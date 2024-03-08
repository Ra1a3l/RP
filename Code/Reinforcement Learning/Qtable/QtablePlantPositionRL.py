import numpy as np
import matplotlib.pyplot as plt

#Careful with the number of episodes as the simulation can take a long time to run.

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


class QLearningAgent:
    def __init__(self, num_actions, num_arm_positions, num_plants):
        """
        Initialize a Q-learning agent with given parameters.

        Parameters:
        num_actions (int): Number of possible actions.
        num_arm_positions (int): Number of positions the arm can move to.
        num_plants (int): Number of plants to measure.
        """
        self.num_actions = num_actions
        self.num_arm_positions = num_arm_positions
        self.num_plants = num_plants

        # Initialize Q-table with zeros
        self.q_table = np.zeros((2**num_plants * num_arm_positions, num_actions), dtype=float)

        # Q-learning parameters
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2

    def state_to_index(self, state):
        """
        Convert the state into an index in the Q-table.

        Parameters:
        state (tuple): Current state consisting of arm position and measurements taken.

        Returns:
        int: Index in the Q-table corresponding to the state.
        """
        arm_position, measurements_taken = state
        # Ensure arm_position and measurements_taken are within the valid range
        arm_position = max(0, min(arm_position, self.num_arm_positions - 1))
        
        # Adjust the index for the plants starting from position 1
        measurements_index = int("".join(map(str, measurements_taken[:])), 2)
        
        return arm_position * (2**(self.num_plants - 1)) + measurements_index

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        Parameters:
        state (tuple): Current state consisting of arm position and measurements taken.

        Returns:
        int: Selected action.
        """
        # Map state to index
        state_index = self.state_to_index(state)

        # Exploration-exploitation trade-off
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)  # Explore
        else:
            return np.argmax(self.q_table[state_index, :])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning update rule.

        Parameters:
        state (tuple): Current state consisting of arm position and measurements taken.
        action (int): Action taken.
        reward (float): Reward received.
        next_state (tuple): Next state after taking the action.
        """
        # Map state and next_state to indices
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)

        old_value = self.q_table[state_index, action]
        next_max = np.max(self.q_table[next_state_index, :])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state_index, action] = new_value


# Instantiate the environment and the Q-learning agent
env = RoboticArmEnvironment(num_positions=5, num_plants=8)
agent = QLearningAgent(num_actions=4, num_arm_positions=5, num_plants=8)

# Training loop
num_episodes = 1000
cumulative_rewards = []
# Calculate the number of steps taken per episode
steps_taken = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    counter = 0

    while counter < 100:
        # Choose an action
        action = agent.select_action(state)

        # Take the chosen action and observe the new state and reward
        next_state, reward, done, _ = env.step(str(action))

        # Update the Q-table
        agent.update_q_table(state, action, reward, next_state)

        total_reward += reward
        state = next_state

        counter += 1
        

        if done:
            print("Done in : ", counter, "steps !!")
            break
    
    steps_taken.append(counter)
        

    final_measurements_taken = env.get_measurements_taken()
    print("Final Measurements Taken:", final_measurements_taken)

    cumulative_rewards.append(total_reward)
    # Print total reward for the episode
    # print(f"Episode {episode + 1}, Total Reward: {total_reward}")

q_table = agent.q_table
print(q_table)
# Plot the cumulative rewards
plt.plot(range(1, num_episodes + 1), cumulative_rewards, label='Cumulative Reward')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.show()

# Evaluate the trained agent
state = env.reset()

# Lists to store recorded actions, states, and rewards
recorded_actions = []
recorded_states = []
recorded_rewards = []
Counter = 0

while Counter < 100:
    action = agent.select_action(state)

    # Record the action, state, and reward
    recorded_actions.append(action)
    recorded_states.append(state)
    recorded_rewards.append(env.perform_action(action))  # Record the actual reward from the environment

    next_state, _, done, _ = env.step(str(action))
    state = next_state

    Counter += 1

    if done:
        print("Done in :", Counter, "steps!!")
        
        final_measurements_taken = env.get_measurements_taken()
        print("Final Measurements Taken:", final_measurements_taken)
        break

# Convert lists to NumPy arrays
recorded_actions_array = np.array(recorded_actions)
recorded_states_array = np.array(recorded_states)
recorded_rewards_array = np.array(recorded_rewards)

# Combine them into a single array
combined_array = np.column_stack((recorded_actions_array, recorded_states_array, recorded_rewards_array))

# Print the combined array
print("Combined Array:")
print(combined_array)

# Calculate the average rewards per episode
average_rewards = [sum(cumulative_rewards[:i+1]) / len(cumulative_rewards[:i+1]) for i in range(len(cumulative_rewards))]

# Plot the average rewards
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), average_rewards, label='Average Reward per Episode', color='blue')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward per Episode')
plt.legend()
plt.show()

# Plot the number of steps taken to complete the task
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), steps_taken, label='Number of Steps Taken', color='green')
plt.xlabel('Episodes')
plt.ylabel('Number of Steps')
plt.title('Number of Steps Taken to Complete the Task')
plt.legend()
plt.show()

# Calculate the average number of steps taken for each episode range
episode_ranges = [100 * i for i in range(1, num_episodes // 100 + 1)]
average_steps_per_range = []

for range_end in episode_ranges:
    average_steps_per_range.append(sum(steps_taken[:range_end]) / range_end)

# Plot the average number of steps taken per episode range
plt.figure(figsize=(10, 5))
plt.plot(episode_ranges, average_steps_per_range, label='Average Steps Taken', color='green')
plt.xlabel('Number of Episodes')
plt.ylabel('Average Number of Steps')
plt.title('Average Number of Steps Taken to Complete the Task')
plt.legend()
plt.show()


# Define ranges for hyperparameters alpha, epsilon, and gamma
alpha_values = np.linspace(0, 1, 11)  # 11 values from 0 to 1
epsilon_values = np.linspace(0, 1, 11)  # 11 values from 0 to 1
gamma_values = np.linspace(0, 1, 11)  # 11 values from 0 to 1

# Initialize variables to store the results for each hyperparameter
results_alpha = np.zeros(len(alpha_values))
results_epsilon = np.zeros(len(epsilon_values))
results_gamma = np.zeros(len(gamma_values))

# Iterate through all values of alpha
for i, alpha in enumerate(alpha_values):
    # Instantiate the Q-learning agent with the current alpha value
    agent = QLearningAgent(num_actions=4, num_arm_positions=6, num_plants=10)
    agent.alpha = alpha

    # Training loop
    cumulative_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        counter = 0

        while counter < 100:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(str(action))
            agent.update_q_table(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            counter += 1

            if done:
                break

        cumulative_rewards.append(total_reward)

    # Evaluate the performance using the average reward
    average_reward = sum(cumulative_rewards) / num_episodes

    # Store the maximum average reward for this alpha value
    results_alpha[i] = average_reward

# Iterate through all values of epsilon
for j, epsilon in enumerate(epsilon_values):
    # Instantiate the Q-learning agent with the current epsilon value
    agent = QLearningAgent(num_actions=4, num_arm_positions=6, num_plants=10)
    agent.epsilon = epsilon

    # Training loop
    cumulative_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        counter = 0

        while counter < 100:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(str(action))
            agent.update_q_table(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            counter += 1

            if done:
                break

        cumulative_rewards.append(total_reward)

    # Evaluate the performance using the average reward
    average_reward = sum(cumulative_rewards) / num_episodes

    # Store the maximum average reward for this epsilon value
    results_epsilon[j] = average_reward

# Iterate through all values of gamma
for k, gamma in enumerate(gamma_values):
    # Instantiate the Q-learning agent with the current gamma value
    agent = QLearningAgent(num_actions=4, num_arm_positions=6, num_plants=10)
    agent.gamma = gamma

    # Training loop
    cumulative_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        counter = 0

        while counter < 100:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(str(action))
            agent.update_q_table(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            counter += 1

            if done:
                break

        cumulative_rewards.append(total_reward)

    # Evaluate the performance using the average reward
    average_reward = sum(cumulative_rewards) / num_episodes

    # Store the maximum average reward for this gamma value
    results_gamma[k] = average_reward

# Plot the results for alpha
plt.figure(figsize=(8, 6))
plt.plot(alpha_values, results_alpha, marker='o', linestyle='-')
plt.xlabel('Alpha')
plt.ylabel('Average Reward')
plt.title('Average Reward vs. Alpha')
plt.grid(True)
plt.show()

# Plot the results for epsilon
plt.figure(figsize=(8, 6))
plt.plot(epsilon_values, results_epsilon, marker='o', linestyle='-')
plt.xlabel('Epsilon')
plt.ylabel('Average Reward')
plt.title('Average Reward vs. Epsilon')
plt.grid(True)
plt.show()

# Plot the results for gamma
plt.figure(figsize=(8, 6))
plt.plot(gamma_values, results_gamma, marker='o', linestyle='-')
plt.xlabel('Gamma')
plt.ylabel('Average Reward')
plt.title('Average Reward vs. Gamma')
plt.grid(True)
plt.show()
