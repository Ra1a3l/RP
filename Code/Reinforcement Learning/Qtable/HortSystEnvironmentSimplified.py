import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras

class HorticultureEnvironment:
    def __init__(self, agent):
        self.state_space = 4
        self.state = np.zeros(4)
        self.action_space = np.arange(9)  # 9 different actions
        self.simulation_time = 60*24
        self.num_days = 60
        self.agent = agent
        
        # Initialize parameters
        self.Tmax = 35.00
        self.Tmin = 10.00
        self.Tob = 17.00
        self.Tou = 24.00
        self.RUE = 4.01
        self.k = 0.70
        self.PTIinit = 0.025
        self.a = 7.55
        self.b = -0.15
        self.c1 = 2.82
        self.c2 = 74.66
        self.A = 0.3
        self.Bd = 18.7
        self.Bn = 8.5
        self.RHmin = 0.6259
        self.RHmax = 0.9398
        self.Light_day = 3.99
        self.Light_night = 0.88
        self.d = 3.5
        
        #Sequence of actions where the agent doesn't change the temperature and light values
        self.temperature = np.zeros(60)
        self.light = np.zeros(60)
        
        #Milestones for the reward function
        self.milestone_1 = True
        self.milestone_2 = True
        self.milestone_3 = True
        self.milestone_4 = True
        self.milestone_5 = True
        self.milestone_6 = True
        self.milestone_7 = True
        self.milestone_8 = True
        self.milestone_9 = True
        self.milestone_10 = True
        
        self.reset()

    def reset(self):
        # Sample initial states randomly within predefined ranges
        initial_temperature = np.random.randint(5, 18) * 2  # Generate random integer between 5 and 17 and multiply by 2
        initial_light = np.random.randint(0, 11) * 10  # Generate random integer between 0 and 10 and multiply by 10
        initial_LAI = 0.1  # Example range for LAI
        initial_day = 0
        
        # Set initial state with randomized values
        self.state = np.array([initial_temperature, initial_light, initial_LAI, initial_day])
        self.hour = 0  # Reset hour 
        self.DMP = 1  # Initial Dry Matter Production
        self.PTI = 1  # Initial Plant Temperature Index
        self.N_uptake = 1
        self.HETc = []
        self.done = False  # Initialize done
        
        #Milestones for the reward function
        self.milestone_1 = True
        self.milestone_2 = True
        self.milestone_3 = True
        self.milestone_4 = True
        self.milestone_5 = True
        self.milestone_6 = True
        self.milestone_7 = True
        self.milestone_8 = True
        self.milestone_9 = True
        self.milestone_10 = True
        
        return self.state

    def step(self, action_index):
        action = self.get_action_from_index(action_index)
        previous_state = np.copy(self.state)  # Store the previous state
        
        self.hort_syst(self.state, action)  # Update the environment for each day
        
        # Calculate reward based on the changes during the day
        reward = self.calculate_reward(previous_state, self.state)
        
        # Store transitions with hindsight goals for HER
        achieved_goal = self.state[2]  # LAI is the goal
        self.agent.remember_HER(previous_state, action_index, reward, self.state, self.done, achieved_goal)
        
        # Increment the day
        self.state[3] += 1
        
        # Reset hour to 0 for the next day
        self.hour = 0
        
        # Check if simulation has ended
        if self.state[3] >= self.num_days:  # Assuming self.num_days represents the total number of days
            self.done = True 
        
        return self.state, reward, self.done, {}

    def calculate_reward(self, previous_state, current_state, target_lai=4, tolerance=0.2):
        # Check if the LAI is within the target range
        if abs(current_state[2] - target_lai) <= tolerance:
            return 1  # Sparse binary reward: +1 for achieving the goal LAI within tolerance
        else:
            return 0   # Sparse binary reward: 0 for all other states


    
    def calculate_reward_old(self, previous_state, current_state):
        reward = 0
        # Check if the LAI has increased
        if (current_state[2] - previous_state[2]) > 0:
            reward += 0#20*(current_state[2]-previous_state[2])  # Positive reward for LAI growth
            
        if (current_state[2] - previous_state[2]) <= 0:
            reward -= 0#10*(current_state[2]-previous_state[2])  # Negative reward for LAI decay or stagnation
            
        
        #try to encourage the agent to stay in a state whith high LAI
        if current_state[2] <= 0.1:
            reward -= 0.01
        if current_state[2] > 0.1 and self.milestone_1:
            reward -= 25
            self.milestone_1 = False
        if current_state[2] > 0.2 and self.milestone_2:
            reward -= 20
            self.milestone_2 = False    
        if current_state[2] > 0.3 and self.milestone_3:
            reward -= 15
            self.milestone_3 = False    
        if current_state[2] > 0.4 and self.milestone_4:
            reward -= 12
            self.milestone_4 = False    
        if current_state[2] > 0.5 and self.milestone_5:
            reward -= 9
            self.milestone_5 = False  
        if current_state[2] > 1 and self.milestone_6:
            reward -= 7
            self.milestone_6 = False            
        if current_state[2] > 2 and self.milestone_7:
            reward -= 5
            self.milestone_7 = False    
        if current_state[2] > 3.5 and self.milestone_8:
            reward -= 3
            self.milestone_8 = False    
        if current_state[2] > 4.5 and self.milestone_9:
            reward -= 1
            self.milestone_9 = False    
        if current_state[2] > 6 and self.milestone_10:
            reward += 50
            self.milestone_10 = False    
        
        # Check if temperature, humidity, or light has changed
        if (current_state[0] != previous_state[0] or
            current_state[1] != previous_state[1]):
            reward -= 0  # Negative reward for changing temperature, humidity, or light
        else:
            reward += 0
        return reward

    def hort_syst(self, state, action):
        
        # Update state based on the chosen action
        #temperature_action, humidity_action, light_action = self.get_action_from_index(action)
        self.temperature_action(action)
        self.light_action(action)
        
        temperature = state[0]
        light = state[1]
        # Simulate 24 hours
        for i in range(24):
            TT = self.thermal_time(temperature)
            #hourly_evapotranspiration = self.ETc(light, temperature, relative_humidity, self.hour)
            
            if self.hour % 24 ==0:
                Delta_PTI = self.delta_PTI(light, TT)
                self.PTI += Delta_PTI
                TT = []
                self.state[2] = self.Leaf_area_index()
                Delta_DMP = self.dmp_change(light)
                self.DMP += Delta_DMP
                Delta_Nup = self.nup_change(light)
                self.N_uptake += Delta_Nup
                #eTc = ET_c(HETc)
                
            self.hour += 1
        
    
    def get_action_from_index(self, action_index):
        """
        Convert action index to the corresponding action.
    
        Parameters:
            action_index (int): Index representing the action.
    
        Returns:
            tuple: Action representing changes in temperature, and light.
        """
        temperature_action = action_index // 3  # 3 actions for temperature
        light_action = action_index % 3  # 3 actions for light
        return temperature_action, light_action

    
    def temperature_action(self, action):
        if action[0] == 0:
            self.state[0] += 0
        if action[0] == 1 and self.state[0] < 32:
            self.state[0] += 2
        if action[0] == 2 and self.state[0] > 13:
            self.state[0] -= 2
            
    def light_action(self, action):
        if action[1] == 0:
            self.state[1] += 0
        if action[1] == 1 and self.state[1] <= 90:
            self.state[1] += 5
        if action[1] == 2 and self.state[1] >= 10:
            self.state[1] -= 5
            
    def PAR(self, R_g):
        return 0.5 * R_g

    def Leaf_area_index(self):
        return self.c1 * self.PTI * self.d / (self.c2 + self.PTI)

    def thermal_time(self, Ta):
        tt = 0
        if Ta < self.Tmin:
            tt = 0
        elif self.Tmin <= Ta < self.Tob:
            tt = (Ta - self.Tmin) / (self.Tob - self.Tmin)
        elif self.Tob <= Ta <= self.Tou:
            tt = 1
        elif self.Tou < Ta <= self.Tmax:
            tt = (self.Tmax - Ta) / (self.Tmax - self.Tou)
        elif Ta > self.Tmax:
            tt = 0
        return tt

    def fi_par(self):
        return 1 - np.exp(-self.k * self.Leaf_area_index())

    def vapor_deficit_pressure(self, T, RH):
        SVP = 610.78 * np.exp((T / (T + 237.3)) * 17.2694)
        return SVP * (1 - RH / 100)

    def ETc(self, rg, T, RH, hour):
        vpd = self.vapor_deficit_pressure(T, RH)
        B = self.aerodynamic_term()
        return self.A * (1 - np.exp(-self.k * self.Leaf_area_index())) * rg + self.Leaf_area_index() * vpd * B

    def dmp_change(self, R_g):
        return self.RUE * self.fi_par() * self.PAR(R_g)

    def ET_c(self, array):
        return sum(array)

    def percent_n(self, R_g):
        DMP = self.dmp_change(R_g)
        return self.a * (DMP ** -self.b)

    def nup_change(self, R_g):
        Percent_n = self.percent_n(R_g)
        DMP = self.dmp_change(R_g)
        return (Percent_n / 100) * DMP

    def delta_PTI(self, rg, TT):
        x = TT / 24 * self.PAR(rg) * self.fi_par()
        return x

    def aerodynamic_term(self):
        return self.Bd if 8 < self.hour % 24 < 20 else self.Bn


class QLearningAgent:
    def __init__(self, num_actions, num_states, exploration_method, temperature):
        
        """
        Initialize a Q-learning agent with given parameters.

        Parameters:
        num_actions (int): Number of possible actions.
        num_states (int): Number of possible states in the environment.
        """
        
        self.num_actions = num_actions
        self.num_states = num_states
        self.exploration_method = exploration_method
        self.temperature = temperature

        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions), dtype=float)

        # Q-learning parameters
        self.alpha = 0.7
        self.gamma = 0.95
        self.epsilon = 0.1
        
        # Define the bins for each continuous variable
        self.temperature_bins = [10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35]  # Define bins for temperature
        self.light_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Define bins for light
        self.LAI_bins = [0,  1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6]  # Define bins for LAI
    
    def select_action(self, state):
        """
        Select an action using the specified exploration method.

        Parameters:
        state (tuple): Current state of the environment.

        Returns:
        int: Selected action.
        """
        if self.exploration_method == 'epsilon-greedy':
            return self.select_action_epsilon_greedy(state)
        elif self.exploration_method == 'boltzmann':
            return self.select_action_boltzmann(state)
        elif self.exploration_method == 'ucb':
            return self.select_action_ucb(state)
        else:
            raise ValueError("Invalid exploration method. Use 'epsilon-greedy' or 'boltzmann'.")
    
    def select_action_epsilon_greedy(self, state):
        """
        Select an action using epsilon-greedy policy.
    
        Parameters:
        state (tuple): Current state of the environment.
    
        Returns:
        int: Selected action.
        """
        # Map state to index
        state_index = self.state_to_index(state)
    
        # Exploration-exploitation trade-off
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)  # Explore
        else:
            # Get the row corresponding to state_index
            row = self.q_table[state_index, :]
            
            # Check if all values are 0
            if np.all(row == 0):
                # If all values are 0, choose a random action
                action = np.random.randint(len(row))
            else:
                # Find indices of maximum values
                max_indices = np.where(row == np.max(row))[0]
                # Choose randomly among the maximum indices
                action = np.random.choice(max_indices)
    
            return action
    
    def select_action_boltzmann(self, state):
        """
        Select an action using Boltzmann exploration.

        Parameters:
        state (tuple): Current state of the environment.

        Returns:
        int: Selected action.
        """
        state_index = self.state_to_index(state)
        row = self.q_table[state_index, :]
        probabilities = np.exp(row / self.temperature) / np.sum(np.exp(row / self.temperature))
        return np.random.choice(range(self.num_actions), p=probabilities)
    
    def select_action_ucb(self, state):
        """
        Select an action using Upper Confidence Bound (UCB).

        Parameters:
        state (tuple): Current state of the environment.

        Returns:
        int: Selected action.
        """
        state_index = self.state_to_index(state)
        ucb_values = np.zeros(self.num_actions)
        total_selections = np.sum(self.num_selections)

        for action in range(self.num_actions):
            if self.num_selections[state_index, action] == 0:
                ucb_values[action] = np.inf
            else:
                average_reward = self.total_rewards[state_index, action] / self.num_selections[state_index, action]
                exploration_bonus = np.sqrt(2 * np.log(total_selections) / self.num_selections[state_index, action])
                ucb_values[action] = average_reward + self.temperature * exploration_bonus

        return np.argmax(ucb_values)
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning update rule.

        Parameters:
        state (tuple): Current state of the environment.
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
        
    def state_to_index(self, state):
        """
        Convert the state into an index in the Q-table.
    
        Parameters:
        state (tuple): Current state of the environment.
    
        Returns:
        int: Index in the Q-table corresponding to the state.
        """
        temperature_index = self.discretize(state[0], self.temperature_bins)
        light_index = self.discretize(state[1], self.light_bins)
        LAI_index = self.discretize(state[2], self.LAI_bins)
        day_value = int(state[3])-1  # Assuming state[3] represents the day directly
    
        # Calculate index using the provided bins
        # Multiply each bin index by a power of 10 to create a unique index
        x = temperature_index + 10 * light_index + 100 * LAI_index + 1000 * day_value
    
        return x
    
    def discretize(self, value, bins):
        """
        Discretize a continuous value into a bin index.
    
        Parameters:
        value (float): Value to discretize.
        bins (list): List of bin edges.
    
        Returns:
        int: Index of the bin that value belongs to.
        """
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                return i
        
        # If value is outside the defined bins, return the index of the last bin
        return len(bins) - 2  # or return len(bins) - 1 if you want to use the last bin
    
    def percentage_filled(self):
        # Count non-zero elements in the Q-table
        non_zero_count = np.count_nonzero(self.q_table)

        # Total number of elements in the Q-table
        total_elements = self.num_states * self.num_actions

        # Calculate percentage filled
        percentage_filled = (non_zero_count / total_elements) * 100

        return percentage_filled


class DeepQNAgent:
    def __init__(self, state_shape, action_shape, learning_rate=0.001, discount_factor=0.95):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Initialize the main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())  # Initialize target model with same weights
        
        # Define replay memory
        self.replay_memory = []
        self.min_replay_size = 1000000
        self.batch_size = 512

    def build_model(self):
        """Builds and returns the deep Q-network model."""
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_shape=self.state_shape, activation='relu', kernel_initializer='he_uniform'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.action_shape, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
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
        for index, (state, action, reward, new_state, done) in enumerate(mini_batch):
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

    def remember(self, state, action, reward, new_state, done):
        """Stores experience in replay memory."""
        self.replay_memory.append((state, action, reward, new_state, done))
        
    def remember_HER(self, state, action, reward, new_state, done, achieved_goal):
        """Stores experience in replay memory with hindsight goals."""
        # Store transition with original goal
        self.replay_memory.append((state, action, reward, new_state, done, 4))
    
        # Store transition with hindsight goals (achieved_goal)
        self.replay_memory.append((state, action, reward, new_state, done, achieved_goal))

            
    def train_HER(self):
        """Trains the agent using experiences from replay memory with hindsight goals."""
        if len(self.replay_memory) < self.min_replay_size:
            return
    
        # Sample mini-batch from replay memory
        mini_batch = random.sample(self.replay_memory, self.batch_size)
    
        # Prepare inputs and targets for training
        X = []
        Y = []
        for (state, action, reward, new_state, done, achieved_goal) in mini_batch:
            current_qs = self.model.predict(np.array([state]))[0]
            future_qs = self.target_model.predict(np.array([new_state]))[0]
    
            # Update Q-value for the action taken
            if not done:
                max_future_q = reward + self.discount_factor * np.max(future_qs)
            else:
                max_future_q = reward
    
            current_qs[action] = max_future_q
    
            X.append(state)
            Y.append(current_qs)
    
            # Add additional transitions with hindsight goals
            X.append(state)
            Y.append(future_qs)
    
        # Train the model
        self.model.fit(np.array(X), np.array(Y), batch_size=self.batch_size, verbose=0, shuffle=True)

# Instantiate the environment and the agent

agent = DeepQNAgent((4,), 9)  # Assuming state shape is (4,) and action shape is 27
env = HorticultureEnvironment(agent)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Training loop
num_episodes = 1000
# Define epsilon parameters
initial_epsilon = 1.0  # Initial exploration rate
final_epsilon = 0.1   # Minimum exploration rate
epsilon_decay_steps = num_episodes  # Number of steps for epsilon to decay to its minimum value

# Calculate epsilon decay rate
epsilon_decay_rate = (initial_epsilon - final_epsilon) / epsilon_decay_steps

# Initialize epsilon
epsilon = initial_epsilon

# Lists to store final LAI values and cumulative rewards
final_lai_values = []
cumulative_rewards = []

epsilon = initial_epsilon  # Initialize epsilon

for episode in range(num_episodes):
    state = env.reset()  # Reset the environment
    done = False
    total_reward = 0
    achieved_goal = None  # Initialize achieved_goal for the episode
    
    while not done:
        # Select an action using epsilon-greedy policy
        action = agent.select_action(state, epsilon)
        
        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)
        
        # Remember the experience
        agent.remember_HER(state, action, reward, next_state, done, achieved_goal)
        
        # Update the achieved_goal based on the final LAI value if the episode is done
        if done:
            # Assuming LAI is at index 2 in the state
            final_lai = next_state[2]
            achieved_goal = 4  # Set the achieved goal to 4 since that's the target LAI
            
            # Calculate reward based on the final LAI value
            reward = env.calculate_reward(state, next_state)
            
            # Remember the final transition with the achieved goal
            agent.remember_HER(state, action, reward, next_state, done, achieved_goal)
        
        # Train the agent
        agent.train_HER()
        
        # Update the target network periodically
        if len(agent.replay_memory) % 10 == 0:
            agent.update_target_model()
        
        total_reward += reward
        state = next_state
    
    # Store final LAI value
    final_lai_values.append(final_lai)  # Use the LAI from the last state of the episode
    cumulative_rewards.append(total_reward)
    
    # Decay epsilon
    if epsilon > final_epsilon:
        epsilon -= epsilon_decay_rate
    
    # Print episode stats
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Final LAI: {final_lai}, Epsilon: {epsilon}")





# Simulate one episode with epsilon = 0
epsilon = 0
state = env.reset()
done = False
episode_lai_values = []
episode_temperature_values = []
episode_light_values = []

while not done:
    action = agent.select_action(state, epsilon)
    next_state, _, done, _ = env.step(action)
    episode_lai_values.append(next_state[2])  # Store LAI
    episode_temperature_values.append(next_state[0])
    episode_light_values.append(next_state[1])
    state = next_state
    
# Plotting
plt.figure(figsize=(12, 10))

# Plot final LAI values during training
plt.subplot(3, 1, 1)
plt.plot(range(1, num_episodes + 1), final_lai_values, marker='o', linestyle='-', color='b')
plt.xlabel('Episode')
plt.ylabel('Final LAI')
plt.title('Final LAI Values During Training')

# Plot LAI values in the epsilon=0 episode
plt.subplot(3, 1, 2)
plt.plot(range(len(episode_lai_values)), episode_lai_values, marker='o', linestyle='-', color='r')
plt.xlabel('Time Step')
plt.ylabel('LAI')
plt.title('LAI Values in Epsilon=0 Episode')

# Plot cumulative rewards during training
plt.subplot(3, 1, 3)
plt.plot(range(1, num_episodes + 1), cumulative_rewards, marker='o', linestyle='-', color='g')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Rewards During Training')

plt.tight_layout()
plt.show()
# Get the weights of the model
model_weights = agent.model.get_weights()

# Print the weights of each layer
for i, layer_weights in enumerate(model_weights):
    print(f"Layer {i} weights:")
    print(layer_weights)

# Plot final LAI values
plt.plot(final_lai_values)
plt.xlabel('Episode')
plt.ylabel('Final LAI')
plt.title('Final LAI over Episodes')
plt.grid(True)
plt.show()


def plot_rewards(num_episodes, cumulative_rewards, mean_cumulative_rewards):
    plt.figure()
    plt.plot(range(1, num_episodes + 1), cumulative_rewards, label='Cumulative Reward')
    plt.plot(range(1, num_episodes + 1), mean_cumulative_rewards, color='r', linestyle='--', label='Mean Cumulative Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()

def plot_LAI(num_episodes, Leaf_area_index_training, mean_LAI_rewards):
    plt.figure()
    plt.plot(range(1, num_episodes + 1), Leaf_area_index_training, label='Leaf Area Index')
    plt.plot(range(1, num_episodes + 1), mean_LAI_rewards, color='r', linestyle='--', label='Mean LAI Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Leaf Area Index')
    plt.legend()
    plt.show()

def plot_smoothed_rewards(num_episodes, cumulative_rewards, window_size=20):
    smoothed_rewards = np.convolve(cumulative_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.figure()
    plt.plot(range(window_size, num_episodes + 1), smoothed_rewards, label='Smoothed Cumulative Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()

def plot_smoothed_LAI(num_episodes, Leaf_area_index_training, window_size=20):
    smoothed_LAI = np.convolve(Leaf_area_index_training, np.ones(window_size)/window_size, mode='valid')
    plt.figure()
    plt.plot(range(window_size, num_episodes + 1), smoothed_LAI, label='Smoothed Leaf Area Index')
    plt.xlabel('Episodes')
    plt.ylabel('Leaf Area Index')
    plt.legend()
    plt.show()

num_actions = 9
num_states = 60000   #To be calculated exactly

epsilon_based_training = 'epsilon-greedy'
boltzmann_based_training = 'boltzmann'

agent = QLearningAgent(num_actions, num_states, exploration_method= epsilon_based_training , temperature=1.0)
agent.epsilon = 0.1
agent.temperature = 3

# Calculate the number of states for each variable
num_temperature_states = len(agent.temperature_bins) - 1
num_light_states = len(agent.light_bins) - 1
num_LAI_states = len(agent.LAI_bins) - 1
num_days = 60  # Assuming each day is a unique state

# Calculate the total number of states
total_states = (
    num_temperature_states *
    num_light_states *
    num_LAI_states *
    num_days
)

print("Total number of states:", total_states)

# Create an instance of the environment
env = HorticultureEnvironment()

# Reset the environment to its initial state
env.reset()
print(env.state)

# Training loop
#Careful with the number of episodes as the simulation can take a long time to run.
#Try with 100 episodes first as this takes already about 20 seconds to run
num_episodes = int(input("Enter the number of episodes for training: "))
cumulative_rewards = []
# Calculate the number of steps taken per episode
steps_taken = []
Leaf_area_index_training = []
q_table_filled_percentage = []  # List to store percentage of Q-table filled

def training_loop_1():
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        
        state = env.state
        
        while True:
            # Choose an action
            action = agent.select_action(state)
           

            # Take the chosen action and observe the new state and reward
            next_state, reward, done, _ = env.step(action)

            # Update the Q-table
            agent.update_q_table(env.state, action, reward, next_state)

            total_reward += reward
            state = next_state
            
            if done:
                break
        
        Leaf_area_index_training.append(env.state[2])
            

        cumulative_rewards.append(total_reward)
        # Print total reward for the episode
        print(f"Episode {episode + 1}, Total LAI: {env.state[2]}")
        
        # Calculate percentage of Q-table filled every 10 episodes
        if (episode + 1) % 10 == 0:
            filled_percentage = agent.percentage_filled()
            q_table_filled_percentage.append(filled_percentage)
        
        # Print episode number every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
        
        #agent.epsilon *= 0.9999
def training_loop_2():
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        state = env.state
        final_LAI = 0
        episode_states_actions = []  # Store the sequence of states and actions in this episode

        while True:
            action = agent.select_action(state)
            next_state= env.step(action)[0]
            done = env.step(action)[2]
            episode_states_actions.append((state, action))
            
            state = next_state
            
            if done:
                # Estimate reward for the last state
                final_LAI = state[3]
                episode_states_actions.append((next_state, None))  # None indicates the end of the episode
                break
        
        # Update Q-table based on the episode's sequence of states and actions
        for i in range(len(episode_states_actions) - 1):
            state, action = episode_states_actions[i]
            next_state, _ = episode_states_actions[i + 1]
            #reward = (state[3]/10)**2
            reward = -1/final_LAI
            agent.update_q_table(state, action, reward, next_state)

        
        Leaf_area_index_training.append(env.state[2])
            

        cumulative_rewards.append(total_reward)
        # Print total reward for the episode
        print(f"Episode {episode + 1}, Total LAI: {env.state[2]}")
        
        # Print episode number every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")

def training_loop_3():
    
    epsilon_decrement_episodes = 1000  # Number of episodes after which to decrement epsilon
    epsilon_decrement_value = 0.1  # Value by which to decrement epsilon
    min_epsilon = 0.1  # Minimum value for epsilon
    
    agent.epsilon = 1
    
    for episode in range(num_episodes):
        env.reset()
        state = env.state
        final_LAI = 0
        episode_states_actions = []  # Store the sequence of states and actions in this episode

        while True:
            action = agent.select_action(state)
            next_state= env.step(action)[0]
            done = env.step(action)[2]
            episode_states_actions.append((state, action))
            
            state = next_state
            
            if done:
                # Estimate reward for the last state
                final_LAI = state[2]
                episode_states_actions.append((next_state, None))  # None indicates the end of the episode
                break
        
        reward = -1/final_LAI
        # Update Q-table based on the episode's sequence of states and actions
        for i in range(len(episode_states_actions) - 1):
            state, action = episode_states_actions[i]
            next_state, _ = episode_states_actions[i + 1]
            #reward = (state[3]/10)**2
            
            agent.update_q_table(state, action, reward, next_state)

        
        Leaf_area_index_training.append(env.state[2])
            

        cumulative_rewards.append(reward)
        # Print total reward for the episode
        print(f"Episode {episode + 1}, Total LAI: {env.state[2]}")
        
        # Decrease epsilon every x number of episodes until it reaches min_epsilon
        if (episode + 1) % epsilon_decrement_episodes == 0 and agent.epsilon > min_epsilon:
            agent.epsilon = max(agent.epsilon - epsilon_decrement_value, min_epsilon)
            print(f"Epsilon decremented to {agent.epsilon}")
        
        # Print episode number every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")

def training_loop_4():
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        
        state = env.state
        
        while True:
            # Choose an action
            action = agent.select_action(state)
           

            # Take the chosen action and observe the new state and reward
            next_state, reward, done, _ = env.step_temperature(action)

            # Update the Q-table
            agent.update_q_table(env.state, action, reward, next_state)

            total_reward += reward
            state = next_state
            
            if done:
                break
        
        Leaf_area_index_training.append(env.state[2])
            

        cumulative_rewards.append(total_reward)
        # Print total reward for the episode
        #print(f"Episode {episode + 1}, Total LAI: {env.state[3]}")
        
        # Print episode number every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")

# Define the options for the training loop along with descriptions
loop_options = {
    1: ("training_loop_1", "Simple training loop, functional reward function"),
    2: ("training_loop_2", "Simple training loop, Final LAI based optimization only"),
    3: ("training_loop_3", "Experimental training loop with decaying epsilon"),
    4: ("training_loop_4", "Customized training loop with user-defined settings")
}

# Display instructions, available options, and descriptions
print("Please choose the version of the training loop:")
for key, (value, description) in loop_options.items():
    print(f"{key}: {value} - {description}")

# Get user input for the version of the training loop
while True:
    loop_version = input("Enter the number corresponding to your choice: ")
    if loop_version.isdigit() and int(loop_version) in loop_options:
        loop_version = int(loop_version)
        break
    else:
        print("Invalid input. Please enter a number corresponding to one of the options.")

# Call the selected training loop
selected_loop = loop_options[loop_version][0]
globals()[selected_loop]()

    
q_table = agent.q_table

# Calculate the mean cumulative reward at each episode
mean_cumulative_rewards = np.zeros(num_episodes)
for i in range(num_episodes):
    mean_cumulative_rewards[i] = np.mean(cumulative_rewards[:i+1])

# Calculate the mean LAI reward at each episode
mean_LAI_rewards = np.zeros(num_episodes)
for i in range(num_episodes):
    mean_LAI_rewards[i] = np.mean(Leaf_area_index_training[:i+1])

#Plotting of the results of the training
plot_rewards(num_episodes, cumulative_rewards, mean_cumulative_rewards)
plot_LAI(num_episodes, Leaf_area_index_training, mean_LAI_rewards)
plot_smoothed_rewards(num_episodes, cumulative_rewards)
plot_smoothed_LAI(num_episodes, Leaf_area_index_training)

# Plot the percentage of Q-table filled
plt.figure()
plt.plot(range(10, num_episodes + 1, 10), q_table_filled_percentage)
plt.xlabel('Episodes')
plt.ylabel('Percentage of Q-table Filled')
plt.title('Percentage of Q-table Filled over Episodes')
plt.show()

# Arrays to store parameter values for plotting
temperature_values = []
light_values = []
humidity_values = []
LAI_values = []


# Test agent with only max values of Q-table
agent.exploration_method = epsilon_based_training
agent.epsilon = 0
agent.temperature = 0.01
env.reset()
state = env.state
# Loop to simulate one episode
for i in range(5000):
    # Generate a random number between 1 and 27
    action = agent.select_action(state)
    
    # Take a step with action index 1 (you can choose any action index here)
    next_state, reward, done, _ = env.step(action)
    
    state = next_state
    
    # Append state parameters to their respective arrays
    temperature_values.append(state[0])
    light_values.append(state[1])
    LAI_values.append(state[2])

    if done:
        print("Done!!!")
        break  # End the loop if the termination condition is met

# Plot temperature
plt.figure(figsize=(10, 6))
plt.plot(temperature_values, label='Temperature')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature')
plt.title('Temperature Variation')
plt.legend()
plt.grid(True)
plt.show()

# Plot light
plt.figure(figsize=(10, 6))
plt.plot(light_values, label='Light')
plt.xlabel('Time (hours)')
plt.ylabel('Light')
plt.title('Light Intensity Variation')
plt.legend()
plt.grid(True)
plt.show()

# Plot LAI
plt.figure(figsize=(10, 6))
plt.plot(LAI_values, label='LAI')
plt.xlabel('Time (hours)')
plt.ylabel('LAI')
plt.title('Leaf Area Index (LAI) Variation')
plt.legend()
plt.grid(True)
plt.show()

