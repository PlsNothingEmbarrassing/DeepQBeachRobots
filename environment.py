from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import gymnasium as gym
import numpy as np


MODEL_VERSION = 3


class BeachCleaningEnv(gym.Env):
    """Pure NumPy beach-cleaning environment for Gymnasium and Tianshou."""

    metadata = {"render_modes": []}

    ACTION_MOVE_UP = 0
    ACTION_MOVE_DOWN = 1
    ACTION_MOVE_LEFT = 2
    ACTION_MOVE_RIGHT = 3
    ACTION_PICK = 4

    CHANNEL_ROBOT = 0
    CHANNEL_RUBBISH = 1
    CHANNEL_OBSTACLE = 2
    CHANNEL_VISITED = 3
    CHANNEL_OBSERVED = 4
    CHANNEL_BOUNDARY = 5

    _MOVES = np.asarray(((0, 1), (0, -1), (-1, 0), (1, 0)), dtype=np.int16)

    def __init__(
        self,
        n_obstacles: int = 8,
        n_small_rubbish: int = 40,
        width: int = 20,
        height: int = 20,
        seed: int | None = None,
        max_steps: int = 500,
        observation_radius: int = 5,
        robot_capacity: int | None = None,
        map_width: int = 20,
        map_height: int = 20,
        randomize_robot_start: bool = True,
        n_obstacles_range: Sequence[int] | None = None,
        n_small_rubbish_range: Sequence[int] | None = None,
        episode_index: int = 0,
    ) -> None:
        integers = {
            "n_obstacles": n_obstacles,
            "n_small_rubbish": n_small_rubbish,
            "width": width,
            "height": height,
            "max_steps": max_steps,
            "observation_radius": observation_radius,
            "map_width": map_width,
            "map_height": map_height,
            "episode_index": episode_index,
        }
        if any(
            not isinstance(value, (int, np.integer)) or isinstance(value, bool)
            for value in integers.values()
        ):
            raise ValueError("counts, dimensions, and step settings must be integers")
        if not isinstance(randomize_robot_start, bool):
            raise ValueError("randomize_robot_start must be a boolean")
        if width <= 0 or height <= 0 or map_width <= 0 or map_height <= 0:
            raise ValueError("grid and map dimensions must be positive")
        if width > map_width or height > map_height:
            raise ValueError("map dimensions must contain the entire environment grid")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if observation_radius < 0:
            raise ValueError("observation_radius must be non-negative")
        if n_obstacles < 0 or n_small_rubbish < 0:
            raise ValueError("agent counts must be non-negative")
        if robot_capacity is not None and (
            not isinstance(robot_capacity, (int, np.integer))
            or isinstance(robot_capacity, bool)
            or robot_capacity < 0
        ):
            raise ValueError("robot_capacity must be a non-negative integer")
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")

        self._n_obstacles_range = self._validate_range(
            "n_obstacles_range", n_obstacles_range
        )
        self._n_small_rubbish_range = self._validate_range(
            "n_small_rubbish_range", n_small_rubbish_range
        )
        maximum_obstacles = (
            self._n_obstacles_range[1]
            if self._n_obstacles_range is not None
            else n_obstacles
        )
        maximum_rubbish = (
            self._n_small_rubbish_range[1]
            if self._n_small_rubbish_range is not None
            else n_small_rubbish
        )
        if 1 + maximum_obstacles + maximum_rubbish > width * height:
            raise ValueError("the grid does not have enough cells for the robot and layout")

        self._base_counts = (int(n_obstacles), int(n_small_rubbish))
        self._grid_size = (int(width), int(height))
        self._robot_capacity_override = (
            None if robot_capacity is None else int(robot_capacity)
        )
        self._seed = seed
        self._episode_index = int(episode_index)
        self._max_steps = int(max_steps)
        self._observation_radius = int(observation_radius)
        self._map_size = (int(map_width), int(map_height))
        self._map_offset = ((map_width - width) // 2, (map_height - height) // 2)
        self._randomize_robot_start = randomize_robot_start

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(
                    0.0, 1.0, shape=(map_width, map_height, 6), dtype=np.float32
                ),
                "mask": gym.spaces.MultiBinary(5),
            }
        )
        self._has_reset = False

    @staticmethod
    def _validate_range(
        name: str, value: Sequence[int] | None
    ) -> tuple[int, int] | None:
        if value is None:
            return None
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
        ):
            raise ValueError(f"{name} must contain an inclusive minimum and maximum")
        minimum, maximum = value
        if any(
            not isinstance(item, (int, np.integer)) or isinstance(item, bool)
            for item in value
        ):
            raise ValueError(f"{name} values must be integers")
        if minimum < 0 or maximum < minimum:
            raise ValueError(f"{name} must satisfy 0 <= minimum <= maximum")
        return int(minimum), int(maximum)

    @property
    def next_episode_index(self) -> int:
        return self._episode_index

    @property
    def robot_position(self) -> tuple[int, int]:
        self._require_reset()
        return int(self._robot_position[0]), int(self._robot_position[1])

    @property
    def robot_capacity(self) -> int:
        self._require_reset()
        return self._robot_capacity

    @property
    def current_environment_config(self) -> dict[str, int]:
        self._require_reset()
        return {
            "n_obstacles": self._episode_obstacles,
            "n_small_rubbish": self._episode_rubbish,
            "width": self._grid_size[0],
            "height": self._grid_size[1],
            "max_steps": self._max_steps,
            "observation_radius": self._observation_radius,
            "map_width": self._map_size[0],
            "map_height": self._map_size[1],
        }

    def _sample_count(
        self,
        configured: int,
        configured_range: tuple[int, int] | None,
        rng: np.random.Generator,
    ) -> int:
        if configured_range is None:
            return configured
        return int(rng.integers(configured_range[0], configured_range[1] + 1))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        del options
        if seed is not None:
            self._seed = int(seed)
            self._episode_index = 0
        episode_seed = (
            None if self._seed is None else int(self._seed) + self._episode_index
        )
        super().reset(seed=episode_seed)
        rng = np.random.default_rng(episode_seed)
        self._episode_obstacles = self._sample_count(
            self._base_counts[0], self._n_obstacles_range, rng
        )
        self._episode_rubbish = self._sample_count(
            self._base_counts[1], self._n_small_rubbish_range, rng
        )
        required_cells = 1 + self._episode_obstacles + self._episode_rubbish
        if required_cells > self._grid_size[0] * self._grid_size[1]:
            raise RuntimeError("sampled layout does not fit in the configured grid")

        if self._randomize_robot_start:
            cells = rng.choice(
                self._grid_size[0] * self._grid_size[1],
                size=required_cells,
                replace=False,
            )
            robot_cell, layout_cells = int(cells[0]), cells[1:]
        else:
            robot_cell = self._fixed_start_cell()
            available = np.delete(
                np.arange(self._grid_size[0] * self._grid_size[1]), robot_cell
            )
            layout_cells = rng.choice(
                available, size=required_cells - 1, replace=False
            )

        width, height = self._grid_size
        self._robot_position = np.asarray(divmod(robot_cell, height), dtype=np.int16)
        self._obstacles = np.zeros((width, height), dtype=bool)
        self._rubbish = np.zeros((width, height), dtype=bool)
        obstacle_cells = layout_cells[: self._episode_obstacles]
        rubbish_cells = layout_cells[self._episode_obstacles :]
        if obstacle_cells.size:
            self._obstacles[np.unravel_index(obstacle_cells, (width, height))] = True
        if rubbish_cells.size:
            self._rubbish[np.unravel_index(rubbish_cells, (width, height))] = True

        self._robot_capacity = (
            self._episode_rubbish
            if self._robot_capacity_override is None
            else self._robot_capacity_override
        )
        self._remaining_rubbish = self._episode_rubbish
        self._step_count = 0
        self._episode_reward = 0.0
        self._picked_count = 0
        self._failed_moves = 0
        self._visited = np.zeros(self._grid_size, dtype=bool)
        self._observed = np.zeros(self._grid_size, dtype=bool)
        self._mark_robot_visited()
        self._observe_local_area()
        self._episode_index += 1
        self._has_reset = True
        return self.get_observation(), {}

    def _fixed_start_cell(self) -> int:
        width, height = self._grid_size
        x = min(width - 1, width // 2 + width // 4)
        y = height // 2
        return x * height + y

    def step(self, action: int):
        self._require_reset()
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be an integer from 0 to 4, got {action!r}")
        action = int(action)
        successful = True
        first_visit = False

        if action == self.ACTION_PICK:
            position = tuple(self._robot_position)
            successful = bool(self._rubbish[position] and self._robot_capacity > 0)
            if successful:
                self._rubbish[position] = False
                self._remaining_rubbish -= 1
                self._picked_count += 1
                self._robot_capacity -= 1
        else:
            target = self._robot_position + self._MOVES[action]
            successful = bool(
                0 <= target[0] < self._grid_size[0]
                and 0 <= target[1] < self._grid_size[1]
                and not self._obstacles[tuple(target)]
            )
            if successful:
                self._robot_position = target
                first_visit = not bool(self._visited[tuple(target)])
            else:
                self._failed_moves += 1

        self._mark_robot_visited()
        self._observe_local_area()
        self._step_count += 1
        reward = self.calculate_reward(action, successful, first_visit)
        self._episode_reward += reward
        terminated = self.is_done()
        truncated = self._step_count >= self._max_steps and not terminated
        info = {"episode": self.episode_metrics.copy()} if terminated or truncated else {}
        return (
            self.get_observation(),
            np.float32(reward),
            terminated,
            truncated,
            info,
        )

    def _observe_local_area(self) -> None:
        x, y = self._robot_position
        radius = self._observation_radius
        self._observed[
            max(0, x - radius) : min(self._grid_size[0], x + radius + 1),
            max(0, y - radius) : min(self._grid_size[1], y + radius + 1),
        ] = True

    def _mark_robot_visited(self) -> None:
        self._visited[tuple(self._robot_position)] = True

    def get_full_observation(self) -> np.ndarray:
        self._require_reset()
        state = np.zeros((*self._grid_size, 3), dtype=np.float32)
        state[tuple(self._robot_position) + (0,)] = 1.0
        state[:, :, 1] = self._rubbish
        state[:, :, 2] = self._obstacles
        return state

    def get_observation(self) -> dict[str, np.ndarray]:
        self._require_reset()
        map_width, map_height = self._map_size
        width, height = self._grid_size
        offset_x, offset_y = self._map_offset
        state = np.zeros((map_width, map_height, 6), dtype=np.float32)
        state[:, :, self.CHANNEL_BOUNDARY] = 1.0
        grid_slice = (
            slice(offset_x, offset_x + width),
            slice(offset_y, offset_y + height),
        )
        state[grid_slice + (self.CHANNEL_BOUNDARY,)] = 0.0
        state[grid_slice + (self.CHANNEL_RUBBISH,)] = self._rubbish & self._observed
        state[grid_slice + (self.CHANNEL_OBSTACLE,)] = self._obstacles & self._observed
        state[grid_slice + (self.CHANNEL_VISITED,)] = self._visited
        state[grid_slice + (self.CHANNEL_OBSERVED,)] = self._observed
        robot_x, robot_y = self._robot_position
        state[offset_x + robot_x, offset_y + robot_y, self.CHANNEL_ROBOT] = 1.0
        mask = np.ones(5, dtype=np.int8)
        mask[self.ACTION_PICK] = int(
            self._rubbish[tuple(self._robot_position)] and self._robot_capacity > 0
        )
        return {"obs": state, "mask": mask}

    def calculate_reward(
        self, action: int, successful: bool, first_visit: bool = False
    ) -> float:
        if action == self.ACTION_PICK:
            reward = 1.0 if successful else -0.2
        elif not successful:
            reward = -0.2
        else:
            reward = -0.02 + (0.02 if first_visit else 0.0)
        if self.is_done():
            reward += 10.0
        return reward

    def is_done(self) -> bool:
        return self._remaining_rubbish == 0

    @property
    def episode_metrics(self) -> dict[str, int | float | bool]:
        self._require_reset()
        return {
            "steps": self._step_count,
            "reward": self._episode_reward,
            "picked": self._picked_count,
            "failed_moves": self._failed_moves,
            "total_rubbish": self._episode_rubbish,
            "completed": self.is_done(),
        }

    def state_dict(self) -> dict:
        """Return the active episode state needed for an exact resume."""
        self._require_reset()
        names = (
            "_episode_obstacles",
            "_episode_rubbish",
            "_robot_position",
            "_obstacles",
            "_rubbish",
            "_robot_capacity",
            "_remaining_rubbish",
            "_step_count",
            "_episode_reward",
            "_picked_count",
            "_failed_moves",
            "_visited",
            "_observed",
        )
        return {
            "seed": self._seed,
            "episode_index": self._episode_index,
            "values": {name: deepcopy(getattr(self, name)) for name in names},
        }

    def load_state_dict(self, state: dict) -> None:
        self._seed = state["seed"]
        self._episode_index = int(state["episode_index"])
        for name, value in state["values"].items():
            setattr(self, name, deepcopy(value))
        self._has_reset = True

    def _require_reset(self) -> None:
        if not self._has_reset:
            raise RuntimeError("Call reset() before using the environment")
