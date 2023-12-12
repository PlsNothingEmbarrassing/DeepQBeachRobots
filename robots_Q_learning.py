from .model import Beach, PICKED, UNPICKED
from .agents import LCRobot, SmallRubbish, Obstacle
import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import gym
from gym import spaces
import numpy as np

HEIGHT = 50
WIDTH = 50


class MesaGymEnv(gym.Env):
    def __init__(self):
        super(MesaGymEnv, self).__init__()
        self.model = Beach(1, 1, 150, WIDTH, HEIGHT)
        self.action_space = spaces.Discrete(
            5
        )  # 4 directions and "pick" to begin with to keep simple
        self.observation_space = spaces.Box(low=0, high=3, shape=(50, 50), dtype=np.int)
        # Storing states for simplified access
        self.previous_state = None
        self.new_state = None

    def reset(self):
        # Reset model with default params and return obs
        self.model = Beach(1, 1, 150, WIDTH, HEIGHT)
        initial_observation = self.get_observation()
        return initial_observation

    def step(self, action):
        # Apply action to model
        self.model.step()
        # Get new observation
        new_observation = self.get_observation()
        # Update states
        self.update_states()
        # Calculate action reward
        reward = self.calculate_reward(action)
        # Check if all rubbish cleaned and sim should end
        done = self.is_done()
        # Gather info for analyss
        info = {}

        return new_observation, reward, done, info

    def update_states(self, new_state):
        self.previous_state = self.new_state
        self.new_state = new_state

    def calculate_reward(self, action):
        # Reward function
        reward = 0
        if action == "pick":
            # Check if pick operation was successful and reward accordingly
            if self.pick_successful(self.previous_state, self.new_state):
                reward += 10
            else:
                reward -= 1

        elif action == "move_up":
            if self.hit_obstacle():
                # Penalise if hit obstacle
                reward -= 5
            else:
                reward += 0.1
        elif action == "move_down":
            if self.hit_obstacle() or self.hit_grid_limit():
                # Penalise if hit obstacle
                reward -= 5
            else:
                reward += 0.1
        elif action == "move_left":
            if self.hit_obstacle():
                # Penalise if hit obstacle
                reward -= 5
            else:
                reward += 0.1
        elif action == "move_right":
            if self.hit_obstacle():
                # Penalise if hit obstacle
                reward -= 5
            else:
                reward += 0.1

        if self.is_done():
            # Big reward if completed all rubbsih
            reward += 50

        return reward

    def pick_successful(self, prev_state, new_state):
        # Check that the number of instances of rubbish has decreased after pick has been called
        mesa_agent = self.get_mesa_agent()
        x, y = mesa_agent.pos
        prev_cell_contents = prev_state[x, y]
        new_cell_contents = new_state[x, y]
        # Check that the number of small rubbish instances has decreased
        prev_rubbish_count = np.sum(prev_cell_contents == 2)
        new_rubbish_count = np.sum(new_cell_contents == 2)
        return prev_rubbish_count > new_rubbish_count

    def get_observation(self):
        # Create 3d array representing the mesa environment (2d to represent grid and another dim for occupying agents)
        # Each cell can represent up to 3 agents as need fixed length for DQL
        grid_state = np.zeros(
            (self.model.grid.width, self.model.grid.height, 3), dtype=int
        )

        # Iterate over each cell in the grid
        for content, (x, y) in self.model.grid.coord_iter():
            if content is not None:
                # Array to for cell content
                cell_contents = np.zeros(3, dtype=int)

                # Fill the cell_contents array based on the agents in the cell
                for i, agent in enumerate(content[:3]):  # Consider up to 3 agents
                    if isinstance(agent, LCRobot):
                        cell_contents[i] = 1  # LCRobot
                    # Picked rubbish will not be considered as it would confuse agent if it had stacked rubbish on it
                    elif isinstance(agent, SmallRubbish) and (agent.state != PICKED):
                        cell_contents[i] = 2  # SmallRubbish
                    elif isinstance(agent, Obstacle):
                        cell_contents[i] = 3  # Obstacle

                # Assign the cell_contents to the grid state
                grid_state[x, y] = cell_contents

        return grid_state

    def is_done(self):
        # Check if any rubbish instances (2) are in grid representation
        for x in range(self.new_state.shape[0]):
            for y in range(self.new_state.shape[1]):
                cell_contents = self.new_state[x, y]
                if 2 in cell_contents:
                    return False
        return True

    def hit_obstacle(self):
        mesa_agent = self.get_mesa_agent()
        x, y = mesa_agent.pos
        cell_contents = self.new_state[x, y]
        # If agent (1) and obstacle (3) are in the same cell then a collision has occured
        if 1 and 3 in cell_contents:
            return True
        else:
            return False

    def get_mesa_agent(self):
        # Returns the LC robot mesa agent
        for agent in self.model.schedule.agents:
            if isinstance(agent, LCRobot):
                return agent
        print("No LC robot in environment")


# Spell "Optimiser" correctly
optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
# Initialise training environment
train_env = MesaGymEnv()
# Params probably need adjustment as there is only 1 state so far
fc_layer_params = (100, 50)
q_net = q_network.QNetwork(
    train_env.observation_space,
    train_env.action_space,
    fc_layer_params=fc_layer_params,
)

agent = dqn_agent.DqnAgent(
    # TODO define train_step_counter
    train_env.time_step_spec(),
    train_env.action_space,
    q_network=q_net,
    optimizer=optimiser,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=None,
)
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_space,
    batch_size=train_env.batch_size,
    max_length=500,
)


def collect_step(environment, agent, replay_buffer):
    time_step = environment.current_time_step()
    action_step = agent.collect_policy.action(time_step)

    new_time_step = environment.step(action_step.action.numpy()[0])
    traj = trajectory.from_transition(time_step, action_step, new_time_step)
