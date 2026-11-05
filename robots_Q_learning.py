"""PyTorch/Tianshou training and evaluation for the beach-cleaning agent."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch, ReplayBuffer
from tianshou.utils.torch_utils import policy_within_training_step
from torch import nn

from .environment import MODEL_VERSION, BeachCleaningEnv


@dataclass(frozen=True)
class DqnConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gradient_clipping: float = 10.0
    target_update_period: int = 500
    batch_size: int = 64
    replay_buffer_capacity: int = 100_000
    initial_collect_steps: int = 5_000
    epsilon_initial: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 40_000
    n_step_update: int = 3
    train_interval: int = 4


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    training_environment: Mapping[str, object]
    evaluation_environment: Mapping[str, object]
    minimum_environment_steps: int
    maximum_environment_steps: int | None
    evaluation_interval: int
    promotion_completion_rate: float = 0.8


COMMON_ENVIRONMENT = {
    "observation_radius": 5,
    "map_width": 20,
    "map_height": 20,
    "randomize_robot_start": True,
}

DEFAULT_ENVIRONMENT = {
    **COMMON_ENVIRONMENT,
    "n_obstacles": 8,
    "n_small_rubbish": 40,
    "width": 20,
    "height": 20,
    "max_steps": 500,
}

DEFAULT_CURRICULUM = (
    CurriculumStage(
        name="stage_1",
        training_environment={
            **COMMON_ENVIRONMENT,
            "n_obstacles": 0,
            "n_small_rubbish": 10,
            "n_small_rubbish_range": (10, 20),
            "width": 10,
            "height": 10,
            "max_steps": 200,
        },
        evaluation_environment={
            **COMMON_ENVIRONMENT,
            "n_obstacles": 0,
            "n_small_rubbish": 10,
            "width": 10,
            "height": 10,
            "max_steps": 200,
        },
        minimum_environment_steps=10_000,
        maximum_environment_steps=25_000,
        evaluation_interval=5_000,
    ),
    CurriculumStage(
        name="stage_2",
        training_environment={
            **COMMON_ENVIRONMENT,
            "n_obstacles": 2,
            "n_obstacles_range": (0, 4),
            "n_small_rubbish": 10,
            "n_small_rubbish_range": (10, 20),
            "width": 10,
            "height": 10,
            "max_steps": 300,
        },
        evaluation_environment={
            **COMMON_ENVIRONMENT,
            "n_obstacles": 2,
            "n_small_rubbish": 20,
            "width": 10,
            "height": 10,
            "max_steps": 300,
        },
        minimum_environment_steps=20_000,
        maximum_environment_steps=50_000,
        evaluation_interval=10_000,
    ),
    CurriculumStage(
        name="stage_3",
        training_environment={
            **COMMON_ENVIRONMENT,
            "n_obstacles": 4,
            "n_obstacles_range": (4, 12),
            "n_small_rubbish": 32,
            "n_small_rubbish_range": (32, 48),
            "width": 20,
            "height": 20,
            "max_steps": 500,
        },
        evaluation_environment=DEFAULT_ENVIRONMENT,
        minimum_environment_steps=0,
        maximum_environment_steps=None,
        evaluation_interval=25_000,
    ),
)

DEFAULT_VALIDATION_SEEDS = tuple(range(10_000, 10_010))
DEFAULT_FINAL_TEST_SEEDS = tuple(range(20_000, 20_050))


class BeachQNetwork(nn.Module):
    """Compact CNN mapping the persistent map to five Q-values."""

    def __init__(self, observation_shape: Sequence[int], action_count: int) -> None:
        super().__init__()
        if len(observation_shape) != 3 or min(observation_shape[:2]) < 7:
            raise ValueError("The two strided 3x3 CNN layers require a map at least 7x7")
        channels = int(observation_shape[2])
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_size = self.features(
                torch.zeros(1, channels, *observation_shape[:2])
            ).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_count),
        )

    def forward(self, obs, state=None, info=None):
        del info
        value = torch.as_tensor(
            obs, dtype=torch.float32, device=next(self.parameters()).device
        )
        if value.ndim == 3:
            value = value.unsqueeze(0)
        value = value.permute(0, 3, 1, 2)
        return self.head(self.features(value)), state


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_dqn_config(config: DqnConfig) -> None:
    if config.learning_rate <= 0 or not 0 <= config.gamma <= 1:
        raise ValueError("learning_rate must be positive and gamma must be between 0 and 1")
    if config.gradient_clipping <= 0 or config.target_update_period <= 0:
        raise ValueError("gradient clipping and target update period must be positive")
    if config.batch_size <= 0 or config.replay_buffer_capacity <= 0:
        raise ValueError("batch size and replay capacity must be positive")
    if config.initial_collect_steps < config.n_step_update + 1:
        raise ValueError("initial_collect_steps must fill at least one n-step sequence")
    if config.n_step_update <= 0 or config.train_interval <= 0:
        raise ValueError("n_step_update and train_interval must be positive")
    if config.replay_buffer_capacity < config.n_step_update + 1:
        raise ValueError("replay buffer is too small for the configured n-step update")
    if config.epsilon_decay_steps <= 0:
        raise ValueError("epsilon_decay_steps must be positive")
    if not 0 <= config.epsilon_final <= config.epsilon_initial <= 1:
        raise ValueError("epsilon values must satisfy 0 <= final <= initial <= 1")


def _device(device: str | torch.device | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def create_agent(
    environment: BeachCleaningEnv,
    config: DqnConfig = DqnConfig(),
    device: str | torch.device | None = None,
):
    """Create the masked three-step Double DQN used by every stage."""
    _validate_dqn_config(config)
    target_device = _device(device)
    observation_space = environment.observation_space["obs"]
    model = BeachQNetwork(
        observation_space.shape, int(environment.action_space.n)
    ).to(target_device)
    policy = DiscreteQLearningPolicy(
        model=model,
        action_space=environment.action_space,
        observation_space=environment.observation_space,
        eps_training=config.epsilon_initial,
        eps_inference=0.0,
    )
    agent = DQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=config.learning_rate),
        gamma=config.gamma,
        n_step_return_horizon=config.n_step_update,
        target_update_freq=config.target_update_period,
        is_double=True,
        huber_loss_delta=1.0,
    ).to(target_device)
    # Ponytail: Tianshou 2.0 does not expose DQN gradient clipping publicly.
    agent.optim._max_grad_norm = config.gradient_clipping
    return agent


def _epsilon(config: DqnConfig, train_steps: int) -> float:
    progress = min(1.0, train_steps / config.epsilon_decay_steps)
    return config.epsilon_initial + progress * (
        config.epsilon_final - config.epsilon_initial
    )


def select_action(
    agent,
    observation: Mapping[str, np.ndarray],
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Select a legal epsilon-greedy action without framework environment wrappers."""
    valid_actions = np.flatnonzero(np.asarray(observation["mask"], dtype=bool))
    if not len(valid_actions):
        raise RuntimeError("environment returned an all-false action mask")
    if epsilon and (rng or np.random.default_rng()).random() < epsilon:
        return int((rng or np.random.default_rng()).choice(valid_actions))
    model = agent.policy.model
    was_training = model.training
    model.eval()
    with torch.no_grad():
        q_values, _ = model(np.expand_dims(observation["obs"], axis=0))
        mask = torch.as_tensor(
            observation["mask"], dtype=torch.bool, device=q_values.device
        )
        action = int(q_values[0].masked_fill(~mask, -torch.inf).argmax().item())
    model.train(was_training)
    return action


def _add_transition(
    replay_buffer: ReplayBuffer,
    observation: Mapping[str, np.ndarray],
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    next_observation: Mapping[str, np.ndarray],
) -> None:
    replay_buffer.add(
        Batch(
            obs=observation,
            act=action,
            rew=reward,
            terminated=terminated,
            truncated=truncated,
            obs_next=next_observation,
        )
    )


def collect_step(
    environment: BeachCleaningEnv,
    observation: Mapping[str, np.ndarray],
    agent,
    replay_buffer: ReplayBuffer,
    epsilon: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    action = select_action(agent, observation, epsilon, rng)
    next_observation, reward, terminated, truncated, _ = environment.step(action)
    _add_transition(
        replay_buffer,
        observation,
        action,
        float(reward),
        terminated,
        truncated,
        next_observation,
    )
    if terminated or truncated:
        next_observation, _ = environment.reset()
    return next_observation


def random_action(
    _environment: BeachCleaningEnv,
    observation: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> int:
    return int(rng.choice(np.flatnonzero(observation["mask"])))


def nearest_known_rubbish_action(
    _environment: BeachCleaningEnv,
    observation: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> int:
    """Move toward the nearest rubbish remembered in the discovered map."""
    if observation["mask"][BeachCleaningEnv.ACTION_PICK]:
        return BeachCleaningEnv.ACTION_PICK
    state = observation["obs"]
    robot_position = np.argwhere(
        state[:, :, BeachCleaningEnv.CHANNEL_ROBOT] > 0
    )[0]
    rubbish_positions = np.argwhere(
        state[:, :, BeachCleaningEnv.CHANNEL_RUBBISH] > 0
    )
    if len(rubbish_positions):
        target = min(
            rubbish_positions,
            key=lambda point: int(np.abs(point - robot_position).sum()),
        )
        dx, dy = target - robot_position
        if abs(int(dx)) >= abs(int(dy)) and dx:
            return (
                BeachCleaningEnv.ACTION_MOVE_RIGHT
                if dx > 0
                else BeachCleaningEnv.ACTION_MOVE_LEFT
            )
        if dy:
            return (
                BeachCleaningEnv.ACTION_MOVE_UP
                if dy > 0
                else BeachCleaningEnv.ACTION_MOVE_DOWN
            )
    return int(rng.integers(BeachCleaningEnv.ACTION_PICK))


def evaluate_action_selector(
    action_selector: Callable[
        [BeachCleaningEnv, Mapping[str, np.ndarray], np.random.Generator], int
    ],
    environment_kwargs: Mapping[str, object],
    seeds: Sequence[int],
) -> dict[str, float]:
    episodes = []
    for seed in seeds:
        environment = BeachCleaningEnv(**environment_kwargs)
        observation, _ = environment.reset(seed=seed)
        rng = np.random.default_rng(seed)
        terminated = truncated = False
        while not (terminated or truncated):
            observation, _, terminated, truncated, _ = environment.step(
                action_selector(environment, observation, rng)
            )
        episodes.append(environment.episode_metrics.copy())
    return _summarize_metrics(episodes)


def evaluate_policy(
    agent, environment_kwargs: Mapping[str, object], seeds: Sequence[int]
):
    return evaluate_action_selector(
        lambda _environment, observation, _rng: select_action(agent, observation),
        environment_kwargs,
        seeds,
    )


def _summarize_metrics(episodes: list[dict]) -> dict[str, float]:
    if not episodes:
        raise ValueError("At least one evaluation episode is required")
    summary = {
        f"average_{key}": float(np.mean([episode[key] for episode in episodes]))
        for key in ("reward", "picked", "failed_moves", "steps", "total_rubbish")
    }
    summary["completion_rate"] = float(
        np.mean([bool(episode["completed"]) for episode in episodes])
    )
    summary["episodes"] = len(episodes)
    return summary


def evaluate_baselines(
    environment_kwargs: Mapping[str, object], seeds: Sequence[int]
):
    return {
        "random": evaluate_action_selector(random_action, environment_kwargs, seeds),
        "nearest_known": evaluate_action_selector(
            nearest_known_rubbish_action, environment_kwargs, seeds
        ),
    }


def _checkpoint_metadata_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).parent / "model_metadata.json"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def _torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _agent_state(agent, config: DqnConfig) -> dict:
    return {
        "model_version": MODEL_VERSION,
        "dqn_config": asdict(config),
        "algorithm": agent.state_dict(),
        "optimizer": agent.optim._optim.state_dict(),
        "algorithm_iteration": agent._iter,
    }


def _restore_agent(agent, saved: Mapping[str, object]) -> None:
    agent.load_state_dict(saved["algorithm"])
    agent.optim._optim.load_state_dict(saved["optimizer"])
    agent._iter = int(saved.get("algorithm_iteration", 0))


def _save_exported_agent(agent, config: DqnConfig, directory: Path) -> str:
    checkpoint_path = directory / "agent.pt"
    _torch_save(checkpoint_path, _agent_state(agent, config))
    _write_json(
        directory / "model_metadata.json",
        {
            "model_version": MODEL_VERSION,
            "framework": "pytorch-tianshou",
            "checkpoint_path": str(checkpoint_path),
            "dqn_config": asdict(config),
        },
    )
    return str(checkpoint_path)


def load_agent_checkpoint(
    checkpoint_path: str | Path,
    environment_kwargs: Mapping[str, object],
    config: DqnConfig = DqnConfig(),
    device: str | torch.device | None = None,
):
    """Recreate a v3 agent and restore a trusted PyTorch checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    metadata_path = _checkpoint_metadata_path(checkpoint_path)
    if not metadata_path.is_file():
        raise ValueError(
            "Legacy TensorFlow checkpoints are incompatible with model version 3; retrain"
        )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("model_version") != MODEL_VERSION:
        raise ValueError("Checkpoint model version is incompatible; retrain")
    if metadata.get("dqn_config") != asdict(config):
        raise ValueError("Checkpoint DQN configuration does not match")
    environment = BeachCleaningEnv(**environment_kwargs)
    agent = create_agent(environment, config, device)
    saved = torch.load(
        checkpoint_path, map_location=_device(device), weights_only=False
    )
    _restore_agent(agent, saved)
    return agent


def _runtime_metadata() -> dict[str, object]:
    package_versions = {}
    for package in ("torch", "tianshou", "gymnasium", "numpy", "mesa"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def _validate_curriculum(
    curriculum: Sequence[CurriculumStage], total_environment_steps: int
) -> None:
    if total_environment_steps <= 0:
        raise ValueError("total_environment_steps must be positive")
    if not curriculum:
        raise ValueError("curriculum must contain at least one stage")
    names = set()
    map_shapes = set()
    for stage in curriculum:
        if not stage.name or stage.name in names:
            raise ValueError("curriculum stage names must be non-empty and unique")
        names.add(stage.name)
        if stage.minimum_environment_steps < 0 or stage.evaluation_interval <= 0:
            raise ValueError("stage step counts and evaluation intervals are invalid")
        if (
            stage.maximum_environment_steps is not None
            and stage.maximum_environment_steps < stage.minimum_environment_steps
        ):
            raise ValueError("stage maximum must be at least its minimum")
        if not 0 <= stage.promotion_completion_rate <= 1:
            raise ValueError("promotion completion rate must be between zero and one")
        for environment in (stage.training_environment, stage.evaluation_environment):
            map_shapes.add(
                (environment.get("map_width", 20), environment.get("map_height", 20))
            )
    if len(map_shapes) != 1:
        raise ValueError("all curriculum environments must share one map shape")


def _normalized_run_config(
    total_environment_steps: int,
    seed: int | None,
    config: DqnConfig,
    curriculum: Sequence[CurriculumStage],
    validation_seeds: Sequence[int],
    final_test_seeds: Sequence[int],
) -> dict[str, object]:
    return json.loads(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "framework": "pytorch-tianshou",
                "total_environment_steps": total_environment_steps,
                "seed": seed,
                "dqn": asdict(config),
                "curriculum": [asdict(stage) for stage in curriculum],
                "validation_seeds": list(validation_seeds),
                "final_test_seeds": list(final_test_seeds),
            },
            sort_keys=True,
        )
    )


def _score(metrics: Mapping[str, float]) -> tuple[float, float, float]:
    return (
        float(metrics["completion_rate"]),
        float(metrics["average_picked"]),
        float(metrics["average_reward"]),
    )


def _save_pickle(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_training_checkpoint(
    directory: Path,
    environment_steps: int,
    agent,
    config: DqnConfig,
    environment: BeachCleaningEnv,
    replay_buffer: ReplayBuffer,
    collector_rng: np.random.Generator,
) -> tuple[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"checkpoint_{environment_steps:09d}.pt"
    replay_path = directory / f"replay_{environment_steps:09d}.pkl"
    saved = _agent_state(agent, config)
    saved.update(
        {
            "environment": environment.state_dict(),
            "collector_rng": collector_rng.bit_generator.state,
            "rng": _capture_rng_state(),
        }
    )
    _torch_save(checkpoint_path, saved)
    _save_pickle(replay_path, replay_buffer)

    checkpoints = sorted(directory.glob("checkpoint_*.pt"))
    for old_checkpoint in checkpoints[:-3]:
        old_replay = directory / old_checkpoint.name.replace("checkpoint_", "replay_").replace(
            ".pt", ".pkl"
        )
        old_checkpoint.unlink(missing_ok=True)
        old_replay.unlink(missing_ok=True)
    return str(checkpoint_path), str(replay_path)


def train(
    total_environment_steps: int = 200_000,
    output_directory: str | Path = "training_runs/curriculum",
    seed: int | None = 7,
    curriculum: Sequence[CurriculumStage] = DEFAULT_CURRICULUM,
    config: DqnConfig = DqnConfig(),
    validation_seeds: Sequence[int] = DEFAULT_VALIDATION_SEEDS,
    final_test_seeds: Sequence[int] = DEFAULT_FINAL_TEST_SEEDS,
    log_interval_updates: int = 1_000,
    resume: bool = False,
    require_gpu: bool = False,
    device: str | torch.device | None = None,
):
    """Train one Tianshou Double DQN through the three-stage curriculum."""
    _validate_dqn_config(config)
    _validate_curriculum(curriculum, total_environment_steps)
    if not validation_seeds or not final_test_seeds:
        raise ValueError("validation and final-test seeds must not be empty")
    if set(validation_seeds) & set(final_test_seeds):
        raise ValueError("validation and final-test seeds must be disjoint")
    if log_interval_updates <= 0:
        raise ValueError("log_interval_updates must be positive")
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("No PyTorch CUDA GPU was detected")
    target_device = _device(device)
    if require_gpu and target_device.type != "cuda":
        raise ValueError("--require-gpu cannot be combined with a CPU device")

    seed_everything(seed)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    run_config = _normalized_run_config(
        total_environment_steps,
        seed,
        config,
        curriculum,
        validation_seeds,
        final_test_seeds,
    )
    run_config_path = output_path / "run_config.json"
    state_path = output_path / "training_state.json"
    if resume:
        if not run_config_path.is_file() or not state_path.is_file():
            raise ValueError("No resumable training state exists in the output directory")
        if json.loads(run_config_path.read_text()) != run_config:
            raise ValueError("Resume configuration does not match the saved run")
        saved_state = json.loads(state_path.read_text())
        if saved_state.get("complete"):
            raise ValueError("The saved training run is already complete")
    else:
        if any(output_path.iterdir()):
            raise ValueError("Output directory is not empty; use --resume or a new directory")
        _write_json(run_config_path, run_config)
        saved_state = None

    stage_index = int(saved_state["stage_index"]) if saved_state else 0
    stage_steps = int(saved_state["stage_steps"]) if saved_state else 0
    environment_steps = int(saved_state["environment_steps"]) if saved_state else 0
    train_steps = int(saved_state["train_steps"]) if saved_state else 0
    warmup_remaining = (
        int(saved_state["warmup_remaining"])
        if saved_state
        else config.initial_collect_steps
    )
    train_interval_progress = (
        int(saved_state["train_interval_progress"]) if saved_state else 0
    )
    promotion_streak = int(saved_state["promotion_streak"]) if saved_state else 0
    evaluations = list(saved_state.get("evaluations", [])) if saved_state else []
    stage_history = list(saved_state.get("stage_history", [])) if saved_state else []
    best_scores = dict(saved_state.get("best_scores", {})) if saved_state else {}
    best_checkpoint_paths = (
        dict(saved_state.get("best_checkpoint_paths", {})) if saved_state else {}
    )
    elapsed_before_resume = (
        float(saved_state.get("elapsed_seconds", 0.0)) if saved_state else 0.0
    )

    if stage_index >= len(curriculum):
        raise ValueError("Saved run is already complete")
    stage = curriculum[stage_index]
    stage_seed = None if seed is None else seed + stage_index * 1_000_000
    environment = BeachCleaningEnv(
        seed=stage_seed,
        episode_index=int(saved_state.get("next_episode_index", 0)) if saved_state else 0,
        **stage.training_environment,
    )
    observation, _ = environment.reset()
    agent = create_agent(environment, config, target_device)
    replay_buffer = ReplayBuffer(
        config.replay_buffer_capacity,
        random_seed=(42 if seed is None else seed + stage_index),
    )
    collector_rng = np.random.default_rng(seed)

    if saved_state:
        checkpoint = torch.load(
            saved_state["checkpoint_path"],
            map_location=target_device,
            weights_only=False,
        )
        if checkpoint.get("model_version") != MODEL_VERSION:
            raise ValueError("Saved checkpoint model version is incompatible; retrain")
        _restore_agent(agent, checkpoint)
        environment.load_state_dict(checkpoint["environment"])
        observation = environment.get_observation()
        collector_rng.bit_generator.state = checkpoint["collector_rng"]
        _restore_rng_state(checkpoint["rng"])
        with Path(saved_state["replay_path"]).open("rb") as handle:
            replay_buffer = pickle.load(handle)

    loss_sum = 0.0
    losses_since_log = 0
    started_at = time.perf_counter()

    def elapsed_seconds() -> float:
        return elapsed_before_resume + time.perf_counter() - started_at

    def save_training_state() -> None:
        checkpoint_path, replay_path = _save_training_checkpoint(
            output_path / "checkpoints",
            environment_steps,
            agent,
            config,
            environment,
            replay_buffer,
            collector_rng,
        )
        _write_json(
            state_path,
            {
                "model_version": MODEL_VERSION,
                "stage_index": stage_index,
                "stage_steps": stage_steps,
                "environment_steps": environment_steps,
                "train_steps": train_steps,
                "warmup_remaining": warmup_remaining,
                "train_interval_progress": train_interval_progress,
                "promotion_streak": promotion_streak,
                "next_episode_index": environment.next_episode_index,
                "checkpoint_path": checkpoint_path,
                "replay_path": replay_path,
                "evaluations": evaluations,
                "stage_history": stage_history,
                "best_scores": best_scores,
                "best_checkpoint_paths": best_checkpoint_paths,
                "elapsed_seconds": elapsed_seconds(),
            },
        )

    while environment_steps < total_environment_steps:
        observation = collect_step(
            environment,
            observation,
            agent,
            replay_buffer,
            _epsilon(config, train_steps),
            collector_rng,
        )
        environment_steps += 1
        stage_steps += 1
        if warmup_remaining:
            warmup_remaining -= 1
        else:
            train_interval_progress += 1
            if train_interval_progress >= config.train_interval:
                with policy_within_training_step(agent.policy):
                    stats = agent.update(
                        replay_buffer, sample_size=config.batch_size
                    )
                train_steps += 1
                loss_sum += float(stats.loss)
                losses_since_log += 1
                train_interval_progress = 0
                if losses_since_log >= log_interval_updates:
                    print(
                        f"environment_steps={environment_steps} "
                        f"train_steps={train_steps} "
                        f"loss={loss_sum / losses_since_log:.4f} "
                        f"device={target_device}"
                    )
                    loss_sum = 0.0
                    losses_since_log = 0

        at_stage_maximum = (
            stage.maximum_environment_steps is not None
            and stage_steps >= stage.maximum_environment_steps
        )
        at_total_maximum = environment_steps >= total_environment_steps
        evaluation_due = (
            stage_steps % stage.evaluation_interval == 0
            or at_stage_maximum
            or at_total_maximum
        )
        if evaluation_due:
            metrics = evaluate_policy(
                agent, stage.evaluation_environment, validation_seeds
            )
            metrics.update(
                {
                    "stage": stage.name,
                    "stage_environment_steps": stage_steps,
                    "environment_steps": environment_steps,
                    "train_steps": train_steps,
                }
            )
            evaluations.append(metrics)
            print(f"evaluation={metrics}")
            stage_score = _score(metrics)
            saved_best = best_scores.get(stage.name)
            if saved_best is None or tuple(saved_best) < stage_score:
                best_scores[stage.name] = list(stage_score)
                best_checkpoint_paths[stage.name] = _save_exported_agent(
                    agent, config, output_path / "best_checkpoints" / stage.name
                )
            if (
                stage_steps >= stage.minimum_environment_steps
                and metrics["completion_rate"] >= stage.promotion_completion_rate
            ):
                promotion_streak += 1
            else:
                promotion_streak = 0
            save_training_state()

        final_stage = stage_index == len(curriculum) - 1
        stage_complete = at_stage_maximum or promotion_streak >= 2
        if final_stage:
            if at_total_maximum or promotion_streak >= 2:
                break
            continue
        if not stage_complete:
            continue

        stage_history.append(
            {
                "stage": stage.name,
                "environment_steps": stage_steps,
                "completed_at_total_step": environment_steps,
                "promoted_by_threshold": promotion_streak >= 2,
            }
        )
        stage_index += 1
        stage = curriculum[stage_index]
        stage_steps = 0
        warmup_remaining = config.initial_collect_steps
        train_interval_progress = 0
        promotion_streak = 0
        stage_seed = None if seed is None else seed + stage_index * 1_000_000
        environment = BeachCleaningEnv(seed=stage_seed, **stage.training_environment)
        observation, _ = environment.reset()
        replay_buffer = ReplayBuffer(
            config.replay_buffer_capacity,
            random_seed=(42 if seed is None else seed + stage_index),
        )
        save_training_state()

    if not stage_history or stage_history[-1].get("stage") != stage.name:
        stage_history.append(
            {
                "stage": stage.name,
                "environment_steps": stage_steps,
                "completed_at_total_step": environment_steps,
                "promoted_by_threshold": promotion_streak >= 2,
            }
        )

    completed_train_steps = train_steps
    final_stage = curriculum[-1]
    final_checkpoint_path = best_checkpoint_paths.get(final_stage.name)
    if final_checkpoint_path is None:
        final_checkpoint_path = _save_exported_agent(
            agent, config, output_path / "best_checkpoints" / final_stage.name
        )
        best_checkpoint_paths[final_stage.name] = final_checkpoint_path
    else:
        best_saved = torch.load(
            final_checkpoint_path, map_location=target_device, weights_only=False
        )
        _restore_agent(agent, best_saved)

    final_test = evaluate_policy(
        agent, final_stage.evaluation_environment, final_test_seeds
    )
    baselines = evaluate_baselines(final_stage.evaluation_environment, final_test_seeds)
    beats_baselines = all(
        _score(final_test) > _score(baseline) for baseline in baselines.values()
    )
    quality_acceptance = {
        "completion_rate_target": 0.8,
        "completion_rate_achieved": final_test["completion_rate"],
        "beats_all_baselines": beats_baselines,
        "passed": final_test["completion_rate"] >= 0.8 and beats_baselines,
    }
    elapsed = elapsed_seconds()
    result = {
        **run_config,
        "runtime": _runtime_metadata(),
        "environment_steps": environment_steps,
        "train_steps": completed_train_steps,
        "elapsed_seconds": elapsed,
        "environment_steps_per_second": environment_steps / elapsed if elapsed else None,
        "stage_history": stage_history,
        "evaluations": evaluations,
        "best_checkpoint_paths": best_checkpoint_paths,
        "best_checkpoint_path": final_checkpoint_path,
        "baselines": baselines,
        "final_test": final_test,
        "quality_acceptance": quality_acceptance,
    }
    _write_json(output_path / "training_metrics.json", result)
    completed_state = json.loads(state_path.read_text())
    run_complete = quality_acceptance["passed"] or environment_steps >= total_environment_steps
    resumable_stage_history = (
        stage_history
        if run_complete
        else [entry for entry in stage_history if entry["stage"] != final_stage.name]
    )
    completed_state.update(
        {
            "complete": run_complete,
            "elapsed_seconds": elapsed,
            "stage_history": resumable_stage_history,
            "best_scores": best_scores,
            "best_checkpoint_paths": best_checkpoint_paths,
        }
    )
    if not run_complete:
        completed_state["promotion_streak"] = 0
    _write_json(state_path, completed_state)
    return agent, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--output", default="training_runs/curriculum")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    _, result = train(
        total_environment_steps=args.steps,
        output_directory=args.output,
        seed=args.seed,
        resume=args.resume,
        require_gpu=args.require_gpu,
        device=None if args.device == "auto" else args.device,
    )
    if not result["quality_acceptance"]["passed"]:
        raise SystemExit("The trained policy did not meet the stage-3 quality target")


if __name__ == "__main__":
    main()
