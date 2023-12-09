from .model import Beach
from .agents import LCRobot, SmallRubbish, Obstacle
import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
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
        self.action_space = spaces.Discrete(5) # 4 directions and "pick" to begin with to keep simple
        self.observation_space = spaces.Box(low=0, high=3, shape=(50, 50), dtype=np.int)

    def reset(self):
        self.model = Beach(1, 1, 150, WIDTH, HEIGHT)
        initial_observation = self.get_observation()
        return initial_observation
    

    def step(self, action):
        self.model.step()

        

    def get_observation(self):
        # Create 2d array representing the mesa environment
        grid_state = np.zeros((self.model.grid.width, self.model.grid.height), dtype=int)
        
        for content, (x, y) in self.model.grid.coord_iter():
            if content is None:
                grid_state[x, y] = 0  # Empty space
            elif isinstance(content, LCRobot):
                grid_state[x, y] = 1  # LCRobot
            elif isinstance(content, SmallRubbish):
                grid_state[x, y] = 2  # SmallRubbish
            elif isinstance(content, Obstacle):
                grid_state[x, y] = 3  # Obstacle

        return grid_state
    
    def calculate_reward():
        pass

optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

train_env = MesaGymEnv()


q_net = q_network.QNetwork(train_env.observation_space(), train_env.action_space, fc_layer_params=fc_layer_params)

agent = dqn_agent.DqnAgent(train_env.time_step_spec(), train_env.action_space, q_network=q_net, optimizer=optimiser, td_errors_loss_fn=common.element_wise_squared_loss, train_step_counter=train_step_counter)
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec = agent.collect_data_space, batch_size=train_env.batch_size, max_length = replay_buffer_capacity)


# Data collection
# ...

# Training loop
# ...