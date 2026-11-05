import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from gymnasium.utils.env_checker import check_env

from DeepQBeachRobots.environment import MODEL_VERSION, BeachCleaningEnv
from DeepQBeachRobots.robots_Q_learning import (
    CurriculumStage,
    DqnConfig,
    create_agent,
    evaluate_baselines,
    evaluate_policy,
    load_agent_checkpoint,
    select_action,
    train,
)


def move_robot(environment, position):
    environment._robot_position = np.asarray(position, dtype=np.int16)
    environment._mark_robot_visited()
    environment._observe_local_area()


class SimulationTests(unittest.TestCase):
    def test_environment_is_valid_gymnasium(self):
        check_env(
            BeachCleaningEnv(
                n_obstacles=1,
                n_small_rubbish=2,
                width=5,
                height=5,
                map_width=7,
                map_height=7,
            )
        )

    def test_dqn_defaults_and_agent_use_three_step_double_dqn(self):
        config = DqnConfig()
        environment = BeachCleaningEnv()
        agent = create_agent(environment, config, device="cpu")

        self.assertEqual(config.train_interval, 4)
        self.assertEqual(config.epsilon_decay_steps, 40_000)
        self.assertEqual(agent.n_step, 3)
        self.assertTrue(agent.is_double)

    def test_randomized_start_and_layout_are_seeded(self):
        first = BeachCleaningEnv(
            n_obstacles=2, n_small_rubbish=3, width=10, height=10, seed=42
        )
        second = BeachCleaningEnv(
            n_obstacles=2, n_small_rubbish=3, width=10, height=10, seed=42
        )
        first.reset()
        second.reset()

        self.assertEqual(first.robot_position, second.robot_position)
        self.assertTrue(np.array_equal(first._obstacles, second._obstacles))
        self.assertTrue(np.array_equal(first._rubbish, second._rubbish))

    def test_environment_places_requested_counts_and_capacity(self):
        environment = BeachCleaningEnv(
            n_obstacles=0,
            n_small_rubbish=1,
            n_obstacles_range=(1, 3),
            n_small_rubbish_range=(4, 6),
            width=10,
            height=10,
            seed=17,
        )
        environment.reset()
        config = environment.current_environment_config

        self.assertEqual(int(environment._obstacles.sum()), config["n_obstacles"])
        self.assertEqual(int(environment._rubbish.sum()), config["n_small_rubbish"])
        self.assertEqual(environment.robot_capacity, config["n_small_rubbish"])

    def test_environment_rejects_invalid_ranges_map_and_placements(self):
        with self.assertRaises(ValueError):
            BeachCleaningEnv(width=21, height=20)
        with self.assertRaises(ValueError):
            BeachCleaningEnv(n_obstacles_range=(3, 1))
        with self.assertRaises(ValueError):
            BeachCleaningEnv(n_small_rubbish=100, width=5, height=5)
        with self.assertRaises(ValueError):
            BeachCleaningEnv(width=5.5)

    def test_robot_movement_stays_in_bounds_and_avoids_obstacles(self):
        environment = BeachCleaningEnv(
            n_obstacles=0, n_small_rubbish=1, width=5, height=5
        )
        environment.reset(seed=1)
        move_robot(environment, (0, 0))
        environment._obstacles[1, 0] = True

        _, left_reward, _, _, _ = environment.step(
            BeachCleaningEnv.ACTION_MOVE_LEFT
        )
        _, right_reward, _, _, _ = environment.step(
            BeachCleaningEnv.ACTION_MOVE_RIGHT
        )
        self.assertEqual(environment.robot_position, (0, 0))
        self.assertAlmostEqual(float(left_reward), -0.2)
        self.assertAlmostEqual(float(right_reward), -0.2)

    def test_persistent_map_observes_locally_and_clears_picked_rubbish(self):
        environment = BeachCleaningEnv(
            n_obstacles=0,
            n_small_rubbish=1,
            width=5,
            height=5,
            map_width=7,
            map_height=7,
            observation_radius=0,
            randomize_robot_start=False,
            seed=3,
        )
        environment.reset()
        rubbish_position = tuple(np.argwhere(environment._rubbish)[0])
        move_robot(environment, rubbish_position)
        observation = environment.get_observation()
        offset_x, offset_y = environment._map_offset
        map_position = (
            offset_x + rubbish_position[0],
            offset_y + rubbish_position[1],
        )

        self.assertEqual(observation["obs"].shape, (7, 7, 6))
        self.assertEqual(
            observation["obs"][0, 0, BeachCleaningEnv.CHANNEL_BOUNDARY], 1
        )
        self.assertEqual(
            observation["obs"][map_position + (BeachCleaningEnv.CHANNEL_RUBBISH,)],
            1,
        )
        self.assertEqual(observation["mask"][BeachCleaningEnv.ACTION_PICK], 1)

        next_observation, reward, terminated, truncated, _ = environment.step(
            BeachCleaningEnv.ACTION_PICK
        )
        self.assertEqual(float(reward), 11.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(
            next_observation["obs"][
                map_position + (BeachCleaningEnv.CHANNEL_RUBBISH,)
            ],
            0,
        )
        self.assertEqual(next_observation["mask"][BeachCleaningEnv.ACTION_PICK], 0)

    def test_discovered_rubbish_remains_known_after_leaving_sensor_range(self):
        environment = BeachCleaningEnv(
            n_obstacles=0,
            n_small_rubbish=1,
            width=5,
            height=5,
            observation_radius=0,
            randomize_robot_start=False,
            seed=4,
        )
        environment.reset()
        rubbish_position = tuple(np.argwhere(environment._rubbish)[0])
        move_robot(environment, rubbish_position)
        known_before = environment.get_observation()["obs"][
            :, :, BeachCleaningEnv.CHANNEL_RUBBISH
        ].copy()
        destination = (0, 0) if rubbish_position != (0, 0) else (4, 4)
        move_robot(environment, destination)

        self.assertTrue(
            np.array_equal(
                known_before,
                environment.get_observation()["obs"][
                    :, :, BeachCleaningEnv.CHANNEL_RUBBISH
                ],
            )
        )

    def test_rewards_and_remaining_counter_termination(self):
        environment = BeachCleaningEnv(
            n_obstacles=0, n_small_rubbish=1, width=5, height=5
        )
        environment.reset()

        self.assertEqual(
            environment.calculate_reward(
                BeachCleaningEnv.ACTION_MOVE_UP, True, True
            ),
            0,
        )
        self.assertEqual(
            environment.calculate_reward(
                BeachCleaningEnv.ACTION_MOVE_UP, True, False
            ),
            -0.02,
        )
        self.assertEqual(
            environment.calculate_reward(
                BeachCleaningEnv.ACTION_MOVE_UP, False, False
            ),
            -0.2,
        )
        self.assertFalse(environment.is_done())
        environment._remaining_rubbish = 0
        self.assertTrue(environment.is_done())

    def test_action_mask_is_respected_by_greedy_and_random_actions(self):
        environment = BeachCleaningEnv(
            n_obstacles=0,
            n_small_rubbish=1,
            width=5,
            height=5,
            observation_radius=0,
            randomize_robot_start=False,
        )
        observation, _ = environment.reset()
        agent = create_agent(environment, DqnConfig(), device="cpu")
        rng = np.random.default_rng(9)

        for epsilon in (0.0, 1.0):
            for _ in range(25):
                self.assertNotEqual(
                    select_action(agent, observation, epsilon, rng),
                    BeachCleaningEnv.ACTION_PICK,
                )

    def test_random_baseline_is_reproducible(self):
        environment = {
            "n_obstacles": 0,
            "n_small_rubbish": 2,
            "width": 5,
            "height": 5,
            "max_steps": 10,
            "observation_radius": 2,
        }
        self.assertEqual(
            evaluate_baselines(environment, [101, 102]),
            evaluate_baselines(environment, [101, 102]),
        )

    def test_smoke_curriculum_exports_reloadable_v3_policy(self):
        environment = {
            "n_obstacles": 0,
            "n_small_rubbish": 2,
            "width": 5,
            "height": 5,
            "max_steps": 10,
            "observation_radius": 2,
        }
        stages = (
            CurriculumStage("first", environment, environment, 0, 6, 6),
            CurriculumStage("final", environment, environment, 0, None, 6),
        )
        config = DqnConfig(
            batch_size=4,
            replay_buffer_capacity=32,
            initial_collect_steps=4,
            epsilon_decay_steps=10,
            train_interval=1,
        )
        with TemporaryDirectory() as directory:
            _, result = train(
                total_environment_steps=12,
                output_directory=directory,
                seed=5,
                curriculum=stages,
                config=config,
                validation_seeds=[301],
                final_test_seeds=[401],
                log_interval_updates=10,
                device="cpu",
            )
            self.assertEqual(result["model_version"], MODEL_VERSION)
            self.assertEqual(result["framework"], "pytorch-tianshou")
            self.assertEqual(result["environment_steps"], 12)
            self.assertEqual(result["train_steps"], 4)
            self.assertEqual(
                [entry["stage"] for entry in result["stage_history"]],
                ["first", "final"],
            )
            self.assertTrue(Path(directory, "training_metrics.json").is_file())
            self.assertTrue(Path(result["best_checkpoint_path"]).is_file())
            metadata = json.loads(
                Path(result["best_checkpoint_path"])
                .parent.joinpath("model_metadata.json")
                .read_text()
            )
            self.assertEqual(metadata["model_version"], MODEL_VERSION)
            loaded_agent = load_agent_checkpoint(
                result["best_checkpoint_path"], environment, config, device="cpu"
            )
            self.assertEqual(
                evaluate_policy(loaded_agent, environment, [501])["episodes"], 1
            )
            with self.assertRaises(ValueError):
                train(
                    total_environment_steps=13,
                    output_directory=directory,
                    seed=5,
                    curriculum=stages,
                    config=config,
                    validation_seeds=[301],
                    final_test_seeds=[401],
                    resume=True,
                    device="cpu",
                )
            with self.assertRaisesRegex(ValueError, "already complete"):
                train(
                    total_environment_steps=12,
                    output_directory=directory,
                    seed=5,
                    curriculum=stages,
                    config=config,
                    validation_seeds=[301],
                    final_test_seeds=[401],
                    resume=True,
                    device="cpu",
                )

    def test_incomplete_early_stop_can_resume_from_replay_checkpoint(self):
        environment = {
            "n_obstacles": 0,
            "n_small_rubbish": 2,
            "width": 5,
            "height": 5,
            "max_steps": 1,
            "observation_radius": 1,
        }
        stages = (
            CurriculumStage("only", environment, environment, 0, None, 4, 0.0),
        )
        config = DqnConfig(
            batch_size=4,
            replay_buffer_capacity=32,
            initial_collect_steps=4,
            epsilon_decay_steps=10,
            train_interval=1,
        )
        with TemporaryDirectory() as directory:
            _, first = train(
                total_environment_steps=12,
                output_directory=directory,
                seed=8,
                curriculum=stages,
                config=config,
                validation_seeds=[601],
                final_test_seeds=[701],
                device="cpu",
            )
            self.assertEqual(first["environment_steps"], 8)
            _, resumed = train(
                total_environment_steps=12,
                output_directory=directory,
                seed=8,
                curriculum=stages,
                config=config,
                validation_seeds=[601],
                final_test_seeds=[701],
                resume=True,
                device="cpu",
            )
            self.assertEqual(resumed["environment_steps"], 12)

    def test_legacy_checkpoint_is_rejected(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Legacy TensorFlow checkpoints"):
                load_agent_checkpoint(
                    Path(directory, "agent"),
                    {
                        "n_obstacles": 0,
                        "n_small_rubbish": 1,
                        "width": 5,
                        "height": 5,
                    },
                    device="cpu",
                )


@unittest.skipUnless(importlib.util.find_spec("mesa"), "Mesa is an optional dependency")
class OptionalMesaTests(unittest.TestCase):
    def test_mesa_model_places_requested_agents(self):
        from DeepQBeachRobots.agents import LCRobot, Obstacle, SmallRubbish
        from DeepQBeachRobots.model import Beach

        model = Beach(1, 2, 3, 10, 10, seed=1)
        agents = list(model.agents)
        self.assertEqual(sum(isinstance(a, LCRobot) for a in agents), 1)
        self.assertEqual(sum(isinstance(a, Obstacle) for a in agents), 2)
        self.assertEqual(sum(isinstance(a, SmallRubbish) for a in agents), 3)


if __name__ == "__main__":
    unittest.main()
