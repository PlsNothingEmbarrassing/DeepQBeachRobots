import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from DeepQBeachRobots.agents import LCRobot, Obstacle, SmallRubbish
from DeepQBeachRobots.environment import MesaGymEnv
from DeepQBeachRobots.model import Beach
from DeepQBeachRobots.robots_Q_learning import (
    DqnConfig,
    evaluate_baselines,
    evaluate_policy,
    load_agent_checkpoint,
    train,
)


def agents_of_type(model, agent_type):
    return [agent for agent in model.schedule.agents if isinstance(agent, agent_type)]


class SimulationTests(unittest.TestCase):
    def test_model_places_requested_agents(self):
        model = Beach(1, 2, 3, 10, 10, seed=1)

        self.assertEqual(len(agents_of_type(model, LCRobot)), 1)
        self.assertEqual(len(agents_of_type(model, Obstacle)), 2)
        self.assertEqual(len(agents_of_type(model, SmallRubbish)), 3)

    def test_model_seed_reproduces_positions(self):
        first = Beach(1, 2, 3, 10, 10, seed=42)
        second = Beach(1, 2, 3, 10, 10, seed=42)

        first_positions = [agent.pos for agent in first.schedule.agents]
        second_positions = [agent.pos for agent in second.schedule.agents]
        self.assertEqual(first_positions, second_positions)

    def test_robot_movement_stays_in_bounds(self):
        model = Beach(1, 0, 0, 5, 5, seed=1)
        robot = agents_of_type(model, LCRobot)[0]
        robot.x, robot.y = 0, 0
        model.grid.move_agent(robot, (0, 0))

        self.assertFalse(robot.move_left())
        robot.advance()
        self.assertEqual(robot.pos, (0, 0))

    def test_robot_does_not_move_into_obstacle(self):
        model = Beach(1, 0, 0, 5, 5, seed=1)
        robot = agents_of_type(model, LCRobot)[0]
        obstacle_position = (robot.x + 1, robot.y)
        obstacle = Obstacle(999, model, obstacle_position)
        model.grid.place_agent(obstacle, obstacle_position)
        model.schedule.add(obstacle)

        self.assertFalse(robot.move_right())
        robot.advance()
        self.assertNotEqual(robot.pos, obstacle.pos)

    def test_robot_picks_rubbish_and_reduces_capacity(self):
        model = Beach(1, 0, 0, 5, 5, seed=1)
        robot = agents_of_type(model, LCRobot)[0]
        rubbish = SmallRubbish(999, robot.pos, model)
        model.grid.place_agent(rubbish, robot.pos)
        model.schedule.add(rubbish)

        self.assertTrue(robot.pick_rubbish())
        self.assertEqual(rubbish.state, 1)
        self.assertEqual(robot.payload, [rubbish.unique_id])
        self.assertEqual(robot.capacity, 29)

    def test_environment_default_capacity_can_clear_configured_rubbish(self):
        environment = MesaGymEnv(n_small_rubbish=40, width=20, height=20)
        self.assertEqual(environment.get_mesa_agent().capacity, 40)

    def test_environment_capacity_override_is_preserved(self):
        environment = MesaGymEnv(n_small_rubbish=40, width=20, height=20, robot_capacity=3)
        self.assertEqual(environment.get_mesa_agent().capacity, 3)

    def test_observation_marks_boundaries_and_visited_cells(self):
        environment = MesaGymEnv(
            n_obstacles=0, n_small_rubbish=0, width=5, height=5, observation_radius=1
        )
        environment.reset()
        robot = environment.get_mesa_agent()
        environment.model.grid.move_agent(robot, (0, 0))
        robot.x, robot.y = 0, 0
        environment._visited.fill(0)
        environment._mark_robot_visited()
        time_step = environment.step(MesaGymEnv.ACTION_MOVE_LEFT)
        self.assertEqual(time_step.observation.shape, (3, 3, 5))
        self.assertEqual(time_step.observation[0, 1, 3], 1.0)
        self.assertEqual(time_step.observation[1, 1, 3], 0.0)
        self.assertEqual(time_step.observation[1, 1, 4], 1.0)

    def test_dqn_defaults_are_stable_training_defaults(self):
        config = DqnConfig()
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.target_update_period, 500)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.initial_collect_steps, 5_000)

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

    def test_smoke_training_saves_reloadable_policy_and_metrics(self):
        environment = {
            "n_obstacles": 0,
            "n_small_rubbish": 2,
            "width": 5,
            "height": 5,
            "max_steps": 10,
            "observation_radius": 2,
        }
        config = DqnConfig(
            batch_size=4,
            replay_buffer_capacity=32,
            initial_collect_steps=8,
            epsilon_decay_steps=10,
        )
        with TemporaryDirectory() as directory:
            _, result = train(
                num_iterations=2,
                output_directory=directory,
                seed=5,
                environment_kwargs=environment,
                config=config,
                evaluation_interval=2,
                evaluation_seeds=[301],
                log_interval=10,
            )
            self.assertTrue(Path(directory, "training_metrics.json").is_file())
            self.assertTrue(Path(f"{result['best_checkpoint_path']}.index").is_file())
            loaded_agent = load_agent_checkpoint(
                result["best_checkpoint_path"], environment, config
            )
            self.assertEqual(evaluate_policy(loaded_agent.policy, environment, [301])["episodes"], 1)


if __name__ == "__main__":
    unittest.main()
