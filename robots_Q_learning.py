"""Training and evaluation utilities for the beach-cleaning DQN baseline."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from .environment import MesaGymEnv


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
    epsilon_decay_steps: int = 50_000


DEFAULT_ENVIRONMENT = {
    "n_obstacles": 8,
    "n_small_rubbish": 40,
    "width": 20,
    "height": 20,
    "max_steps": 500,
    "observation_radius": 5,
}


def seed_everything(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


def create_agent(environment: MesaGymEnv, config: DqnConfig = DqnConfig()):
    """Create the CNN DQN used by both the script and notebook."""
    if min(environment.observation_spec.shape[:2]) < 5:
        raise ValueError("The two 3x3 CNN layers require observation_radius >= 2")
    epsilon_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config.epsilon_initial,
        decay_steps=config.epsilon_decay_steps,
        end_learning_rate=config.epsilon_final,
    )
    network = q_network.QNetwork(
        environment.observation_spec,
        environment.action_spec,
        conv_layer_params=((32, (3, 3), 1), (64, (3, 3), 1)),
        fc_layer_params=(128,),
    )
    train_step_counter = tf.Variable(0, dtype=tf.int64, name="train_step")
    agent = dqn_agent.DqnAgent(
        environment.time_step_spec,
        environment.action_spec,
        q_network=network,
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        epsilon_greedy=lambda: epsilon_schedule(train_step_counter),
        gamma=config.gamma,
        target_update_period=config.target_update_period,
        td_errors_loss_fn=common.element_wise_huber_loss,
        gradient_clipping=config.gradient_clipping,
        train_step_counter=train_step_counter,
    )
    agent.initialize()
    return agent


def collect_step(environment: MesaGymEnv, agent, replay_buffer):
    """Collect one transition and return the resulting time step."""
    time_step = environment.current_time_step()
    batched_time_step = tf.nest.map_structure(
        lambda value: tf.expand_dims(value, axis=0), time_step
    )
    action_step = agent.collect_policy.action(batched_time_step)
    next_time_step = environment.step(action_step.action)
    replay_buffer.add_batch(
        trajectory.from_transition(
            batched_time_step,
            action_step,
            tf.nest.map_structure(lambda value: tf.expand_dims(value, axis=0), next_time_step),
        )
    )
    return next_time_step


def _policy_action(policy, time_step) -> int:
    batched_time_step = tf.nest.map_structure(
        lambda value: tf.expand_dims(value, axis=0), time_step
    )
    return int(policy.action(batched_time_step).action.numpy()[0])


def random_action(_environment, _time_step, rng: np.random.Generator) -> int:
    return int(rng.integers(MesaGymEnv.ACTION_PICK + 1))


def nearest_visible_rubbish_action(environment: MesaGymEnv, time_step, rng: np.random.Generator) -> int:
    """A local-observation baseline; it wanders when no rubbish is visible."""
    observation = np.asarray(time_step.observation)
    radius = environment._observation_radius
    if observation[radius, radius, 1] > 0:
        return MesaGymEnv.ACTION_PICK
    rubbish_positions = np.argwhere(observation[:, :, 1] > 0)
    if len(rubbish_positions):
        x, y = min(
            rubbish_positions,
            key=lambda point: abs(int(point[0]) - radius) + abs(int(point[1]) - radius),
        )
        if abs(int(x) - radius) >= abs(int(y) - radius):
            return MesaGymEnv.ACTION_MOVE_RIGHT if x > radius else MesaGymEnv.ACTION_MOVE_LEFT
        return MesaGymEnv.ACTION_MOVE_UP if y > radius else MesaGymEnv.ACTION_MOVE_DOWN
    return int(rng.integers(MesaGymEnv.ACTION_PICK))


def evaluate_action_selector(
    action_selector: Callable[[MesaGymEnv, object, np.random.Generator], int],
    environment_kwargs: Mapping[str, object],
    seeds: list[int],
) -> dict[str, float]:
    episodes = []
    for seed in seeds:
        environment = MesaGymEnv(seed=seed, **environment_kwargs)
        rng = np.random.default_rng(seed)
        time_step = environment.reset()
        while not time_step.is_last():
            time_step = environment.step(action_selector(environment, time_step, rng))
        episodes.append(environment.episode_metrics.copy())
    return _summarize_metrics(episodes, int(environment_kwargs["n_small_rubbish"]))


def evaluate_policy(policy, environment_kwargs: Mapping[str, object], seeds: list[int]):
    return evaluate_action_selector(
        lambda _environment, time_step, _rng: _policy_action(policy, time_step),
        environment_kwargs,
        seeds,
    )


def _summarize_metrics(episodes: list[dict[str, float]], rubbish_count: int) -> dict[str, float]:
    if not episodes:
        raise ValueError("At least one evaluation episode is required")
    summary = {
        f"average_{key}": float(np.mean([episode[key] for episode in episodes]))
        for key in ("reward", "picked", "failed_moves", "steps")
    }
    summary["completion_rate"] = float(np.mean([episode["picked"] == rubbish_count for episode in episodes]))
    summary["episodes"] = len(episodes)
    return summary


def evaluate_baselines(environment_kwargs: Mapping[str, object], seeds: list[int]):
    return {
        "random": evaluate_action_selector(random_action, environment_kwargs, seeds),
        "nearest_visible": evaluate_action_selector(nearest_visible_rubbish_action, environment_kwargs, seeds),
    }


def _unused_policy_path(output_path: Path, iteration: int) -> Path:
    candidate = output_path / f"best_checkpoint_step_{iteration}"
    suffix = 1
    while candidate.exists():
        candidate = output_path / f"best_checkpoint_step_{iteration}_{suffix}"
        suffix += 1
    return candidate


def load_agent_checkpoint(
    checkpoint_path: str | Path,
    environment_kwargs: Mapping[str, object],
    config: DqnConfig = DqnConfig(),
):
    """Recreate a trained agent from an exported TensorFlow checkpoint."""
    environment = MesaGymEnv(**environment_kwargs)
    agent = create_agent(environment, config)
    tf.train.Checkpoint(agent=agent).restore(str(checkpoint_path)).expect_partial()
    return agent


def train(
    num_iterations: int = 200_000,
    output_directory: str | Path = "policies",
    seed: int | None = 7,
    environment_kwargs: Mapping[str, object] | None = None,
    config: DqnConfig = DqnConfig(),
    evaluation_interval: int = 10_000,
    evaluation_seeds: list[int] | None = None,
    log_interval: int = 1_000,
):
    """Train a DQN and persist the best held-out policy and experiment metadata."""
    seed_everything(seed)
    env_config = {**DEFAULT_ENVIRONMENT, **(environment_kwargs or {})}
    evaluation_seeds = evaluation_seeds or list(range(10_000, 10_030))
    environment = MesaGymEnv(seed=seed, **env_config)
    environment.reset()
    agent = create_agent(environment, config)
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec, batch_size=1, max_length=config.replay_buffer_capacity
    )
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=tf.data.AUTOTUNE, sample_batch_size=config.batch_size, num_steps=2
    ).prefetch(tf.data.AUTOTUNE)
    iterator = iter(dataset)
    for _ in range(config.initial_collect_steps):
        collect_step(environment, agent, replay_buffer)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_score: tuple[float, float, float] | None = None
    best_checkpoint_path: str | None = None
    losses: list[float] = []
    for iteration in range(1, num_iterations + 1):
        collect_step(environment, agent, replay_buffer)
        experience, _ = next(iterator)
        loss = float(agent.train(experience).loss)
        losses.append(loss)
        if iteration % log_interval == 0:
            print(f"step={iteration} loss_100={np.mean(losses[-100:]):.4f}")
        if iteration % evaluation_interval == 0 or iteration == num_iterations:
            metrics = evaluate_policy(agent.policy, env_config, evaluation_seeds)
            metrics["iteration"] = iteration
            history.append(metrics)
            score = (metrics["completion_rate"], metrics["average_picked"], metrics["average_reward"])
            print(f"evaluation={metrics}")
            if best_score is None or score > best_score:
                checkpoint_directory = _unused_policy_path(output_path, iteration)
                checkpoint_directory.mkdir()
                checkpoint_path = tf.train.Checkpoint(agent=agent).write(
                    str(checkpoint_directory / "agent")
                )
                best_score = score
                best_checkpoint_path = checkpoint_path

    result = {
        "config": asdict(config), "environment": env_config, "seed": seed,
        "iterations": num_iterations, "evaluation_seeds": evaluation_seeds,
        "baselines": evaluate_baselines(env_config, evaluation_seeds),
        "evaluations": history, "best_checkpoint_path": best_checkpoint_path,
    }
    (output_path / "training_metrics.json").write_text(json.dumps(result, indent=2))
    return agent, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200_000)
    parser.add_argument("--output", default="policies")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluation-interval", type=int, default=10_000)
    args = parser.parse_args()
    train(args.iterations, args.output, args.seed, evaluation_interval=args.evaluation_interval)


if __name__ == "__main__":
    main()
