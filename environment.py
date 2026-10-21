from __future__ import annotations

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .agents import LCRobot, Obstacle, PICKED, SmallRubbish
from .model import Beach


class MesaGymEnv(py_environment.PyEnvironment):
    """TF-Agents environment backed by the beach simulation."""

    ACTION_MOVE_UP = 0
    ACTION_MOVE_DOWN = 1
    ACTION_MOVE_LEFT = 2
    ACTION_MOVE_RIGHT = 3
    ACTION_PICK = 4

    def __init__(
        self,
        n_lc_robots=1,
        n_obstacles=1,
        n_small_rubbish=1500,
        width=50,
        height=50,
        seed=None,
        max_steps=2_000,
        observation_radius=5,
        robot_capacity=None,
    ):
        super().__init__()
        if n_lc_robots != 1:
            raise ValueError("MesaGymEnv currently supports exactly one controlled robot")
        if robot_capacity is None:
            robot_capacity = n_small_rubbish
        if robot_capacity < 0:
            raise ValueError("robot_capacity must be non-negative")
        self._config = (
            n_lc_robots,
            n_obstacles,
            n_small_rubbish,
            width,
            height,
        )
        self._robot_capacity = robot_capacity
        self._seed = seed
        self._episode_index = 0
        self._max_steps = max_steps
        self._observation_radius = observation_radius
        self._step_count = 0
        self._episode_reward = 0.0
        self._picked_count = 0
        self._failed_moves = 0
        self.model = self._new_model()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2 * observation_radius + 1, 2 * observation_radius + 1, 5),
            dtype=np.float32,
            minimum=0,
            maximum=1,
            name="observation",
        )
        self._previous_state = None
        self._visited = np.zeros((width, height), dtype=np.float32)
        self._mark_robot_visited()
        self._new_state = self.get_observation()

    def _new_model(self):
        episode_seed = None
        if self._seed is not None:
            episode_seed = self._seed + self._episode_index
        return Beach(*self._config, seed=episode_seed, robot_capacity=self._robot_capacity)

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def observation_spec(self):
        return self._observation_spec

    @property
    def time_step_spec(self):
        return ts.time_step_spec(self._observation_spec)

    def _reset(self):
        self.model = self._new_model()
        self._episode_index += 1
        self._step_count = 0
        self._episode_reward = 0.0
        self._picked_count = 0
        self._failed_moves = 0
        self._previous_state = None
        self._visited = np.zeros(
            (self.model.grid.width, self.model.grid.height), dtype=np.float32
        )
        self._mark_robot_visited()
        self._new_state = self.get_observation()
        return ts.restart(self._new_state)

    def _step(self, action):
        if self._current_time_step.is_last():
            return self.reset()

        action = int(np.asarray(action).item())
        if not (
            self.action_spec.minimum <= action <= self.action_spec.maximum
        ):
            raise ValueError(f"Action must be an integer from 0 to 4, got {action!r}")

        robot = self.get_mesa_agent()
        successful = True
        if action == self.ACTION_MOVE_UP:
            successful = robot.move_up()
        elif action == self.ACTION_MOVE_DOWN:
            successful = robot.move_down()
        elif action == self.ACTION_MOVE_LEFT:
            successful = robot.move_left()
        elif action == self.ACTION_MOVE_RIGHT:
            successful = robot.move_right()
        elif action == self.ACTION_PICK:
            successful = robot.pick_rubbish()
            if successful:
                self._picked_count += 1
        if action != self.ACTION_PICK and not successful:
            self._failed_moves += 1

        robot.move_payload()
        robot.advance()
        self.model.step()
        self._mark_robot_visited()
        self._previous_state = self._new_state
        self._new_state = self.get_observation()
        self._step_count += 1

        reward = self.calculate_reward(action, successful)
        done = self.is_done() or self._step_count >= self._max_steps
        self._episode_reward += reward
        observation = tf.convert_to_tensor(self._new_state, dtype=tf.float32)
        reward_tensor = tf.convert_to_tensor(reward, dtype=tf.float32)
        if done:
            self._current_time_step = ts.termination(observation, reward_tensor)
        else:
            self._current_time_step = ts.transition(
                observation, reward_tensor, discount=np.float32(1.0)
            )
        return self._current_time_step

    def get_full_observation(self):
        grid_state = np.zeros(
            (self.model.grid.width, self.model.grid.height, 3), dtype=np.float32
        )
        for content, (x, y) in self.model.grid.coord_iter():
            for agent in content:
                if isinstance(agent, LCRobot):
                    grid_state[x, y, 0] = 1
                elif isinstance(agent, SmallRubbish) and agent.state != PICKED:
                    grid_state[x, y, 1] = 1
                elif isinstance(agent, Obstacle):
                    grid_state[x, y, 2] = 1
        return grid_state

    def get_observation(self):
        full_state = self.get_full_observation()
        robot = self.get_mesa_agent()
        radius = self._observation_radius
        boundary = np.zeros(full_state.shape[:2], dtype=np.float32)
        state_with_context = np.concatenate(
            (full_state, boundary[..., np.newaxis], self._visited[..., np.newaxis]), axis=-1
        )
        padded = np.pad(
            state_with_context,
            ((radius, radius), (radius, radius), (0, 0)),
            mode="constant",
        )
        padded[:, :, 3] = 1.0
        padded[radius:-radius, radius:-radius, 3] = 0.0
        x, y = robot.pos
        return padded[
            x : x + 2 * radius + 1,
            y : y + 2 * radius + 1,
            :,
        ].astype(np.float32, copy=False)

    def _mark_robot_visited(self):
        robot = self.get_mesa_agent()
        self._visited[robot.pos] = 1.0

    def calculate_reward(self, action, successful):
        if action == self.ACTION_PICK:
            reward = 10.0 if successful else -1.0
        else:
            reward = -0.2 if successful else -2.0
        if self.is_done():
            reward += 100.0
        return reward

    def is_done(self):
        return not any(
            isinstance(agent, SmallRubbish) and agent.state != PICKED
            for agent in self.model.schedule.agents
        )

    @property
    def episode_metrics(self):
        return {
            "steps": self._step_count,
            "reward": self._episode_reward,
            "picked": self._picked_count,
            "failed_moves": self._failed_moves,
        }

    def get_mesa_agent(self):
        for agent in self.model.schedule.agents:
            if isinstance(agent, LCRobot):
                return agent
        raise RuntimeError("The environment has no LCRobot agent")

    @property
    def previous_state(self):
        return self._previous_state

    @property
    def new_state(self):
        return self._new_state
