import numpy as np
import random
from collections import deque

from HortSystEnvironmentSimplified import HortSystSimplified

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt


class DeepQNAgent:
    def __init__(self, state_shape, action_shape, learning_rate=0.00001, discount_factor=0.95):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Initialize the main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())  # Initialize target model with same weights
        
        # Define replay memory
        self.replay_memory = deque(maxlen=1000000)
        self.min_replay_size = 600
        self.batch_size = 120

    def build_model(self):
        """Builds and returns the deep Q-network model."""
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_shape=self.state_shape, activation='relu', kernel_initializer='he_uniform'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.action_shape, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def select_action(self, state, epsilon):
        """Selects an action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_shape)  # Explore: select a random action
        else:
            q_values = self.model.predict(state.reshape([1, *self.state_shape]))[0]
            return np.argmax(q_values)  # Exploit: select action with highest Q-value

    def train(self):
        """Trains the agent using experiences from replay memory."""
        if len(self.replay_memory) < self.min_replay_size:
            return

        # Sample mini-batch from replay memory
        mini_batch = random.sample(self.replay_memory, self.batch_size)
        
        # Prepare inputs and targets for training
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []
        for index, (state, action, reward, new_state, done, goal) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + self.discount_factor * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = max_future_q

            X.append(state)
            Y.append(current_qs)

        # Train the model
        self.model.fit(np.array(X), np.array(Y), batch_size=self.batch_size, verbose=0, shuffle=True)

    def update_target_model(self):
        """Updates the target model with weights from the main model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, new_state, done, goal):
        """Stores experience in replay memory."""
        self.replay_memory.append((state, action, reward, new_state, done, goal))

    def hindsight_replay(self, achieved_goal):
        """Perform hindsight experience replay."""
        for idx, transition in enumerate(self.replay_memory):
            state, action, reward, new_state, done, _ = transition
            # Replace the goal with the achieved goal
            new_reward = self.calculate_reward(new_state[2], achieved_goal)
            self.replay_memory[idx] = (state, action, new_reward, new_state, done, achieved_goal)


    def calculate_reward(self, new_state, goal):
        # Check if the LAI is within the target range
        if new_state >= goal:
            return 1  # Sparse binary reward: +1 for achieving the goal LAI within tolerance
        else:
            return 0   # Sparse binary reward: 0 for all other states

# Initialize the agent
state_shape = (4,)  # Example state shape
action_shape = 9  # Example action shape
agent = DeepQNAgent(state_shape, action_shape)

# Initialize the environment
env = HortSystSimplified()

# Initialize lists to store results
last_lai_values = []
episode_rewards = []

# Lists to store data
average_lai_values = []
success_rate = []

# Training loop
num_episodes = 200
for episode in range(num_episodes):
    state = env.reset()
    goal = 4  # Fixed goal value
    done = False
    total_reward = 0
    while not done:
        if episode > 10:
            epsilon = max(0.1, 1 - (episode) / 200)  # Epsilon-greedy exploration strategy
        else:
            epsilon = 1
        
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        
        agent.remember(state, action, reward, next_state, done, goal)
        # Perform hindsight experience replay using the final LAI value as the achieved goal
        agent.hindsight_replay(state[2])
        
        
        state = next_state
        total_reward += reward
        
        if state[2] >= goal - 0.2:
            done = True
        
        # Print episode information
        if done:
            print("Episode:", episode + 1, "| Final LAI:", state[2], "| Reward:", total_reward)
        
        agent.train()

    # Calculate success for the episode
    success = 1 if abs(state[2] - goal) <= env.tolerance else 0

    # Update lists
    average_lai_values.append(state[2])  # Append last LAI value for the episode
    success_rate.append(success)  # Append success for the episode

    # Update target network periodically
    if episode % 10 == 0:
        agent.update_target_model()
    
# Calculate moving averages
window_size = 10  # Choose a suitable window size for smoothing
average_lai_values = np.convolve(average_lai_values, np.ones(window_size)/window_size, mode='valid')
smoothed_success_rate = np.convolve(success_rate, np.ones(window_size)/window_size, mode='valid')

# Plot average LAI value over episodes
plt.figure()
plt.plot(average_lai_values)
plt.title('Average LAI Value Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Average LAI Value')
plt.show()

# Plot success rate over episodes
success_rate = [sum(success_rate[:i+1]) / (i+1) for i in range(len(success_rate))]  # Calculate cumulative success rate
plt.figure()
plt.plot(success_rate)
plt.title('Success Rate Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.show()

# Function to run simulations and plot results
def run_simulations(agent, num_simulations=20):
    lai_values = []
    success_rate = []
    for _ in range(num_simulations):
        state = env.reset()
        done = False
        while not done:
            # Select action with epsilon = 0
            action = agent.select_action(state, epsilon=0)
            next_state, reward, done, _ = env.step(action)
            state = next_state

        # Calculate success for the simulation
        success = 1 if state[2] > 3.8 else 0

        # Update lists
        lai_values.append(state[2])
        success_rate.append(success)

    # Plot LAI values for simulations
    plt.figure()
    plt.plot(lai_values)
    plt.title('LAI Values for Simulations')
    plt.xlabel('Simulation')
    plt.ylabel('LAI Value')
    plt.show()

    # Calculate and plot success rate
    success_rate = sum(success_rate) / num_simulations
    print("Success Rate:", success_rate)

# Run simulations with the trained agent
run_simulations(agent)
