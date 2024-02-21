import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

class RoboticArmEnvironment:
    def __init__(self, num_positions, num_plants):
        """
        Initialize the robotic arm environment with given parameters.

        Parameters:
        num_positions (int): Number of positions the arm can move to.
        num_plants (int): Number of plants to measure.
        """
        self.num_positions = num_positions
        self.num_plants = num_plants
        self.arm_position = 0
        self.measurements_taken = np.zeros(num_plants, dtype=int)

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
        tuple: Current arm position and measurements taken.
        """
        self.arm_position = 0
        self.measurements_taken = np.zeros(self.num_plants, dtype=int)
        return self.arm_position, np.copy(self.measurements_taken)

    def take_measurement_left(self):
        """
        Attempt to take a measurement on the left side of the arm position.

        Returns:
        int: Reward obtained from the action.
        """
        plant_position = self.arm_position - 1
        
        if  (self.arm_position > 0) and (plant_position < self.num_plants) and (self.measurements_taken[plant_position] == 0):
            self.measurements_taken[plant_position] = 1
            reward = 20
            return reward  # Measurement successful
        else:
            reward = -10
            return reward # Measurement failed (plant already measured or no measurements left)
        
    def take_measurement_right(self):
        """
        Attempt to take a measurement on the right side of the arm position.

        Returns:
        int: Reward obtained from the action.
        """
        plant_position = self.arm_position + self.num_plants//2 - 1
        
        if (self.arm_position > 0) and (plant_position < self.num_plants) and (self.measurements_taken[plant_position] == 0):
            self.measurements_taken[plant_position] = 1
            reward = 20
            return reward  # Measurement successful
        else:
            reward = -10
            return reward # Measurement failed (plant already measured or no measurements left)

    def move_up(self):
        """
        Move the arm up by one position.

        Returns:
        int: Reward obtained from the action.
        """
        if (self.arm_position < self.num_positions):
            self.arm_position += 1
        reward = -5
        return reward

    def move_down(self):
        """
        Move the arm down by one position.

        Returns:
        int: Reward obtained from the action.
        """
        if (self.arm_position > 0):
            self.arm_position -= 1
        reward = -5
        return reward

    def perform_action(self, action):
        """
        Perform the specified action.

        Parameters:
        action (str): Action to perform.

        Returns:
        int: Reward obtained from the action.
        """
        if action == '0':
            return self.take_measurement_left()
        elif action == '1':
            return self.take_measurement_right()
        elif action == '2':
            return self.move_up()
        elif action == '3':
            return self.move_down()

    def get_state(self):
        """
        Get the current state of the environment.

        Returns:
        tuple: Current arm position and measurements taken.
        """
        return self.arm_position, np.copy(self.measurements_taken)

    def get_measurements_taken(self):
        """
        Get the measurements taken in the environment.

        Returns:
        numpy.ndarray: Copy of the measurements taken.
        """
        return np.copy(self.measurements_taken)

    def step(self, action):
        """
        Execute a step in the environment.

        Parameters:
        action (str): Action to perform.

        Returns:
        tuple: Observation, reward, done flag, additional information.
        """
        reward = self.perform_action(action)
        done = np.all(self.measurements_taken) and (self.arm_position == 0)
        return self.get_state(), reward, done, {}

# Define Deep Q-Network (DQN)
class DQN(models.Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(24, input_dim=state_size, activation='relu')
        self.dense2 = layers.Dense(24, activation='relu')
        self.output_layer = layers.Dense(action_size, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        q_values = self.output_layer(x)
        return q_values

# Define Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        print("Sampled experiences:", samples)
        return zip(*samples)



# Define Agent
class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=32, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        arm_position, measurements_taken = state
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict([np.array([arm_position]), measurements_taken.reshape(1, -1)])
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        arm_position, measurements_taken = state
        next_arm_position, next_measurements_taken = next_state
        self.memory.add((arm_position, measurements_taken, action, reward, next_arm_position, next_measurements_taken, done))


    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return
        experiences = self.memory.sample(self.batch_size)
        states, measurements, actions, rewards, next_states, next_measurements, dones = zip(*experiences)
        
        # Convert tuples of states into arrays
        states = np.array(states)
        measurements = np.array(measurements)
        next_states = np.array(next_states)
        next_measurements = np.array(next_measurements)
        
        target = rewards + (1 - dones) * self.gamma * np.amax(self.target_model.predict([next_states, next_measurements]), axis=1)
        target_full = self.model.predict([states, measurements])
        target_full[np.arange(self.batch_size), actions] = target
        self.model.train_on_batch([states, measurements], target_full)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Define hyperparameters and environment
num_positions = 10
num_plants = 5
state_size = num_plants + 1  # Not sure for this one
action_size = 4  # 4 actions: take measurement left, take measurement right, move up, move down
buffer_size = 10000
batch_size = 32
num_episodes = 1000
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# Initialize environment and agent
env = RoboticArmEnvironment(num_positions, num_plants)
agent = DQNAgent(state_size, action_size, buffer_size, batch_size, gamma, epsilon, epsilon_min, epsilon_decay)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    #state = np.reshape(state, [1, state_size])
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(str(action))
        #next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.replay()
    print("Episode: {}, Total Reward: {}".format(episode+1, total_reward))

# Save trained model
agent.save("dqn_model.h5")
