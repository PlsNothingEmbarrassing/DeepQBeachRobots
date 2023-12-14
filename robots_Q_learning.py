import mesa
import random
import itertools
import math
import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory, TimeStep
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.specs import BoundedTensorSpec, array_spec
from tf_agents.environments import py_environment
from tf_agents.specs import tensor_spec
import os

import numpy as np

PICKED = 1
UNPICKED = 0
HEIGHT = 50
WIDTH = 50


class Beach(mesa.Model):
    def __init__(
        self,
        n_lc_robots,
        n_obstacles,
        n_small_rubbish,
        width,
        height,
    ):
        self.n_lc_robots = n_lc_robots
        self.n_obstacles = n_obstacles
        self.n_small_rubbish = n_small_rubbish
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.grid = mesa.space.MultiGrid(width, height, False)
        self.place_agents()
        self.running = True

    def place_agents(self):
        all_spaces = [
            (x, y) for x in range(self.grid.width) for y in range(self.grid.height)
        ]
        # Use counter to increment ids based on num agents
        counter = itertools.count()

        lc_r_positions = self.find_centre_positions()
        available_spaces = list(filter(lambda x: x not in lc_r_positions, all_spaces))
        search_spaces = self.assign_cleaning_area()
        # Agent is in new position each time to learn
        random.shuffle(lc_r_positions)
        # Creating and placing LC robots
        for i in range(self.n_lc_robots):
            if lc_r_positions:
                u_id = next(counter)
                position = random.randint(0, 9)
                x, y = lc_r_positions.pop(position)
                agent = LCRobot(u_id, (x, y), self)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)
                print("Placed LC at postion: " + str((x, y)))

        # Using shuffle so agent placement should be different each time
        random.shuffle(available_spaces)

        # Creating and placing obstacles
        for i in range(self.n_obstacles):
            if available_spaces:
                u_id = next(counter)
                x, y = available_spaces.pop()
                agent = Obstacle(u_id, self, (x, y))
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)
                # print("Placed obstacle at position: " + str((x,y)))
            else:
                print("No available spaces to place agents")

            # Creating and placing large rubbish
        for i in range(self.n_small_rubbish):
            if available_spaces:
                u_id = next(counter)
                x, y = available_spaces.pop()
                agent = SmallRubbish(u_id, (x, y), self)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)
                # print("Placed SR at position: " + str((x,y)))
            else:
                print("No available spaces to place agents")

    def assign_cleaning_area(self):
        ca_width = math.ceil(self.grid.width / self.n_lc_robots)
        ca_height = self.grid.height
        search_spaces = []
        for i in range(self.n_lc_robots):
            # Get starting positions for each robot search area
            start_x = i * ca_width
            start_y = 0
            # Calculating end positions
            end_x = min(start_x + ca_width, self.grid.width - 1)
            end_y = min(start_y + ca_height, self.grid.height - 1)
            search_space = (start_x, start_y, end_x, end_y)
            search_spaces.append(search_space)
        return search_spaces

    def find_centre_positions(self):
        # Placing lc robots in a circle around centre so they are not placed on top of each other
        centre_x = self.grid.width // 2
        centre_y = self.grid.height // 2
        positions = []
        for i in range(10):
            angle = 2 * math.pi * (i / self.n_lc_robots)
            offset_x = int(centre_x + math.cos(angle) * (self.grid.width // 4))
            offset_y = int(centre_y + math.sin(angle) * (self.grid.height // 4))
            # Make sure that agents are within grid
            offset_x = max(0, min(self.grid.width - 1, offset_x))
            offset_y = max(0, min(self.grid.height - 1, offset_y))
            positions.append((offset_x, offset_y))
        return positions

    def step(self):
        self.schedule.step()


class SmallRubbish(mesa.Agent):
    def __init__(self, id, pos, model, state=UNPICKED, weight=1):
        super().__init__(id, model)
        self.x, self.y = pos
        self.state = state
        self.id = id
        self.model = model
        self.weight = weight

    def __str__(self):
        return "Small Rubbish"


class Obstacle(mesa.Agent):
    def __init__(self, id, model, pos):
        super().__init__(id, model)
        self.x, self.y = pos
        self.id = id
        self.model = model

    def __str__(self):
        return "Obstacle"


class Robot(mesa.Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.id = unique_id
        self.model = model
        self.x, self.y = pos
        self.next_x, self.next_y = self.x, self.y
        self.payload = []

    @property
    def isFull(self):
        return self.capacity == 0

    def wait(self):
        self.next_x = self.x
        self.next_y = self.y

    def advance(self):
        # Advances robot position
        self.x = self.next_x
        self.y = self.next_y

    def move(self):
        self.x = self.next_x
        self.y = self.next_y

    def get_payload_agent(self):
        all_agents = self.model.schedule.agents
        payload_agents = []
        for agent in all_agents:
            if agent.id not in self.payload:
                pass
            else:
                payload_agents.append(agent)
        return payload_agents


class LCRobot(Robot):
    def __init__(
        self,
        id,
        pos,
        model,
    ):
        super().__init__(id, model, pos)
        self.id = id
        self.x, self.y = pos
        self.model = model
        self.next_x, self.next_y = self.x, self.y
        # self.state = state
        # self.capacity = capacity
        self.payload = []
        self.action = None

    @property
    def isFull(self):
        return super().isFull

    def advance(self):
        self.model.grid.move_agent(self, (self.next_x, self.next_y))
        self.x = self.next_x
        self.y = self.next_y

    def wait(self):
        return super().wait()

    # def deliberate(self):
    #     if self.state == EVADING:
    #         return "evade_obstacle"
    #     elif self.state == START:
    #         self.state = MOVING_TO_AREA
    #         return "move_to_area"
    #     elif self.isFull and self.pos != BIN_LOC:
    #         return "move_to_bin"
    #     elif self.isFull and self.pos == BIN_LOC:
    #         return "emptying"

    #     elif self.state == MOVING_TO_AREA:
    #         return "move_to_area"
    #     elif self.state == BUSY:
    #         return "move_through_area"
    #     elif self.state == PICKING:
    #         return "pick"

    #     elif self.state == FINISHED:
    #         return "finished"

    def step(self):
        action = self.action
        if action == "move_up":
            self.move_up()
            self.move_payload()
            self.advance()
        elif action == "move_down":
            self.move_down()
            self.move_payload()
            self.advance()
        elif action == "move_left":
            self.move_left()
            self.move_payload()
            self.advance()
        elif action == "move_right":
            self.move_right()
            self.move_payload()
            self.advance()
        elif action == "pick":
            self.pick_rubbish()
            self.move_payload()
            self.advance()

    def move_payload(self):
        payload_agents = self.get_payload_agent()
        for agent in payload_agents:
            agent.model.grid.move_agent(agent, (self.next_x, self.next_y))
            agent.pos = self.next_x, self.next_y

    def move_up(self):
        # Move agent up
        if self.y < 49:
            self.next_y += 1
            self.next_x = self.x
            pos = (self.next_x, self.next_y)
            self.model.grid.move_agent(self, pos)

    def move_down(self):
        # Move agent down
        if self.y > 0:
            self.next_y -= 1
            self.next_x = self.x
            pos = (self.next_x, self.next_y)
            self.model.grid.move_agent(self, pos)

    def move_left(self):
        # Move agent left
        if self.x > 0:
            self.next_x -= 1
            self.next_y = self.y
            pos = (self.next_x, self.next_y)
            self.model.grid.move_agent(self, pos)

    def move_right(self):
        # Move agent right
        if self.x < 49:
            self.next_x += 1
            self.next_y = self.y
            pos = (self.next_x, self.next_y)
            self.model.grid.move_agent(self, pos)

    # def move_to_bin(self):
    #     target = BIN_LOC
    #     if self.pos != target:
    #         target_x, target_y = target
    #         dx = target_x - self.x
    #         dy = target_y - self.y
    #         if abs(dx) > 0:
    #             self.next_x += dx // abs(dx)
    #         if abs(dy) > 0:
    #             self.next_y += dy // abs(dy)
    #     else:
    #         self.state = EMPTYING
    #         self.drop_off()

    def pick_rubbish(self):
        # Used to get the litter at current position with radius = 0
        litters = self.model.grid.get_neighbors(
            self.pos, moore=False, include_center=True, radius=0
        )

        if litters != None:
            for litter in litters:
                # Check if agent is an Small rubbish instance and has not been picked
                if isinstance(litter, SmallRubbish) and litter.state == UNPICKED:
                    # print("litter picked up")
                    self.payload.append(litter.id)
                    litter.state = PICKED
                    # self.modify_capacity(litter.weight)
                    # self.state = BUSY

        else:
            pass

    # def check_capacity(self, litter_item) -> bool:
    #     return self.capacity - litter_item.weight > 0

    # def modify_capacity(self, weight):
    #     self.capacity -= weight

    def check_cell(self, pos):
        x, y = pos
        if (x > 49) or (y > 49):
            return
        neighbors = self.model.grid.get_neighbors(
            pos, moore=False, include_center=True, radius=0
        )
        if neighbors:
            for neighbor in neighbors:
                if isinstance(neighbor, Obstacle):
                    return "Obstacle"
                elif isinstance(neighbor, Robot) and neighbor != self:
                    return "Robot"
        else:
            return "advance"
        return "advance"

    def get_payload_agent(self):
        return super().get_payload_agent()

    def set_action(self, action):
        self.action = action


class MesaGymEnv(py_environment.PyEnvironment):
    def __init__(self):
        super(MesaGymEnv, self).__init__()
        self.model = Beach(1, 1, 1500, WIDTH, HEIGHT)
        num_actions = 5
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=int, minimum=0, maximum=num_actions - 1, name="action"
        )  # 4 directions and "pick" to begin with to keep simple

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(50, 50, 3), dtype=int, minimum=0, maximum=3, name="observation"
        )
        self._time_step_spec = ts.time_step_spec(self._observation_spec)
        # Storing states for simplified access
        self.previous_state = None
        self.new_state = None

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def observation_spec(self):
        return self._observation_spec

    @property
    def time_step_spec(self) -> ts.TimeStep:
        return ts.time_step_spec(self._observation_spec)

    def _reset(self):
        # Reset model with default params and return obs
        self.model = Beach(1, 1, 1500, WIDTH, HEIGHT)
        initial_observation = self.get_observation()
        self._current_time_step = ts.restart(initial_observation)
        return self._current_time_step

    def _step(self, action):
        if self._current_time_step.is_last():
            return self.reset()
        # Give action to mesa agent in model
        mesa_agent = self.get_mesa_agent()
        if action == 0:
            mesa_agent.move_up()
        elif action == 1:
            mesa_agent.move_down()
        elif action == 2:
            mesa_agent.move_left()
        elif action == 3:
            mesa_agent.move_right()
        elif action == 4:
            mesa_agent.pick_rubbish()
        # Apply these methods regardless of action
        mesa_agent.move_payload()
        mesa_agent.advance()
        # Apply action to model
        self.model.step()
        # Get new observation
        new_observation = self.get_observation()
        print("New observation shape:", new_observation.shape)
        # Update states
        self.update_states(new_observation)
        # Calculate action reward
        reward = self.calculate_reward(action)
        # Check if all rubbish cleaned and sim should end
        done = self.is_done()
        observation_tensor = tf.convert_to_tensor(new_observation, dtype=tf.int32)
        reward = tf.convert_to_tensor(reward, dtype=tf.int32)
        discount = tf.constant(1 if not done else 0, dtype=tf.int32)
        if done:
            self._current_time_step = ts.termination(observation_tensor, reward)
        else:
            self._current_time_step = ts.transition(observation_tensor, reward)
        return self._current_time_step

    def update_states(self, new_state):
        # Update state after step
        self.previous_state = self.new_state
        self.new_state = new_state

    def calculate_reward(self, action):
        # Reward function
        reward = 0
        if action == 4:
            # Check if pick operation was successful and reward accordingly
            if self.pick_successful(self.previous_state, self.new_state):
                reward += 100
            else:
                reward -= 10

        elif action == 0:
            if self.hit_obstacle() or self.move_unsuccessful():
                # Penalise if hit obstacle or hit grid limit
                reward -= 50
            else:
                reward += 1
        elif action == 1:
            if self.hit_obstacle() or self.move_unsuccessful():
                # Penalise if hit obstacle or hit grid limit
                reward -= 50
            else:
                reward += 1
        elif action == 2:
            if self.hit_obstacle() or self.move_unsuccessful():
                # Penalise if hit obstacle or hit grid limit
                reward -= 50
            else:
                reward += 1
        elif action == 3:
            if self.hit_obstacle() or self.move_unsuccessful():
                # Penalise if hit obstacle or hit grid limit
                reward -= 50
            else:
                reward += 1

        if self.is_done():
            # Big reward if completed all rubbsih
            reward += 500

        return reward

    def pick_successful(self, prev_state, new_state):
        # Check that the number of instances of rubbish has decreased after pick has been called
        mesa_agent = self.get_mesa_agent()
        x, y = mesa_agent.pos

        if np.all(prev_state) != None:
            prev_cell_contents = prev_state[x, y]
            new_cell_contents = new_state[x, y]
            # Check that the number of small rubbish instances has decreased
            prev_rubbish_count = np.sum(prev_cell_contents == 2)
            new_rubbish_count = np.sum(new_cell_contents == 2)
            return prev_rubbish_count > new_rubbish_count
        else:
            return False

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
        # Normalise
        # observation = grid_state.astype(np.float32)/3.0

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
        if 1 in cell_contents and 3 in cell_contents:
            return True
        else:
            return False

    def move_unsuccessful(self):
        # Movement is unsuccessful if it hits the grid limits
        return np.array_equal(self.previous_state, self.new_state)

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
print("Action Spec:", train_env.action_spec)
print("Observation Spec:", train_env.observation_spec)
# Reset Environment and retrieve states
train_env.reset()
# Params probably need adjustment as there is only 1 state so far
fc_layer_params = (50, 25)

q_net = q_network.QNetwork(
    train_env.observation_spec,
    train_env.action_spec,
    fc_layer_params=fc_layer_params,
)

train_step_counter = tf.Variable(0)
# Initialise DQN agent
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec,
    train_env.action_spec,
    q_network=q_net,
    optimizer=optimiser,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
)
batch_size = 1
agent.initialize()
# Initialise replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=batch_size,
    max_length=500,
)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
).prefetch(3)
iter_obj = iter(dataset)


def collect_step(environment, agent, replay_buffer):
    # get current time step
    time_step = environment.current_time_step()
    time_step = ts.TimeStep(
        step_type=tf.expand_dims(time_step.step_type, axis=0),
        reward=tf.expand_dims(time_step.reward, axis=0),
        discount=tf.expand_dims(time_step.discount, axis=0),
        observation=tf.expand_dims(time_step.observation, axis=0),
    )
    # Get action
    action_step = agent.collect_policy.action(time_step)

    new_time_step = environment.step(action_step.action.numpy())

    new_observation = tf.expand_dims(new_time_step.observation, axis=0)
    # Add batch dimension
    new_time_step = ts.TimeStep(
        step_type=np.expand_dims(new_time_step.step_type, axis=0),
        reward=np.expand_dims(new_time_step.reward, axis=0),
        discount=np.expand_dims(new_time_step.discount, axis=0),
        observation=new_observation,
    )
    # Create trajectory and store
    traj = trajectory.from_transition(time_step, action_step, new_time_step)
    print("Trajectory observation shape: ", traj.observation.shape)
    print("Trajectory reward shape: ", traj.reward.shape)
    print("Trajectory discount shape: ", traj.discount.shape)
    replay_buffer.add_batch(traj)


def save_policy(policy, directory):
    tf.compat.v2.saved_model.save(policy, directory)


num_iterations = 20000
collect_steps_per_iter = 100
update_steps_per_iter = 10


policy_directory = "policies"
# if not os.path.exists(policy_directory):
#     os.makedirs(policy_directory)
# for _ in range(num_iterations):
#     # Collect steps and save to replay buffer
#     for _ in range(collect_steps_per_iter):
#         collect_step(train_env, agent, replay_buffer)

#     experience, unused_info = next(iter_obj)
#     train_loss = agent.train(experience).loss

#     # Log training loss
#     step = agent.train_step_counter.numpy()
#     if step % 100 == 0:
#         print(f"Step: {step}, Loss: {train_loss}")
#     if step % 100 == 0:
#         # Save policies
#         policy_path = os.path.join(policy_directory, f"policy_step_{step}")
#         save_policy(agent.policy, policy_path)

#     for _ in range(update_steps_per_iter):
#         experience, _ = next(iter_obj)
#         agent.train(experience)


def evaluate_policy(env, policy, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        time_step = env.reset()
        time_step = ts.TimeStep(
            step_type=tf.expand_dims(time_step.step_type, axis=0),
            reward=tf.expand_dims(time_step.reward, axis=0),
            discount=tf.expand_dims(time_step.discount, axis=0),
            observation=tf.expand_dims(time_step.observation, axis=0),
        )
        episode_reward = 0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action.numpy()[0])
            episode_reward += time_step.reward.numpy()
        total_reward += episode_reward

    average_reward = float(total_reward) / num_episodes
    print(f"Average reward: {average_reward}{os.linesep} Total Reward: {total_reward}")

    action_step = agent.collect_policy.action(time_step)


new_env = MesaGymEnv()
loaded_policy = tf.compat.v2.saved_model.load("policies\policy_step_12200")
evaluate_policy(new_env, loaded_policy)
