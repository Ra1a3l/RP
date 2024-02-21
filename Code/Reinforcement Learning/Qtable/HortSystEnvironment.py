import numpy as np
import matplotlib.pyplot as plt

class HorticultureEnvironment:
    def __init__(self):
        self.state_space = 1750
        self.state = np.zeros(5)
        self.action_space = np.arange(27)  # 27 different actions
        self.simulation_time = 60*24
        
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
        
        self.reset()

    def reset(self):
        self.state = np.array([25, 70, 100, 0.1, 0])  # Initial state
        self.hour = 0  # Reset hour 
        self.DMP = 2  # Initial Dry Matter Production
        self.PTI = 10  # Initial Plant Temperature Index
        self.N_uptake = 0.1
        self.HETc = []
        self.done = False  # Initialize done

    def step(self, action_index):
        action = self.get_action_from_index(action_index)
        previous_state = np.copy(self.state)  # Store the previous state
        self.hort_syst(self.state, action)
        reward = self.calculate_reward(previous_state, self.state)
        self.hour += 1  # Increment hour
        self.state[4] = (self.hour - 1) // 24 + 1   #Represents the day to give a sense of time to the state
        #print(self.state[4])
        if self.hour >= self.simulation_time:
            self.done = True  # Placeholder for termination condition
        return self.state, reward, self.done, {}
    
    def calculate_reward(self, previous_state, current_state):
        reward = 0
        # Check if the LAI has increased
        if current_state[3] > previous_state[3]:
            reward += 20  # Positive reward for LAI growth
            
        if current_state[3] < previous_state[3]:
            reward -= 20  # Negative reward for LAI decay
            
        if np.abs(current_state[0]) < 25 and np.abs(current_state[0]) > 20:
            reward += 5
        
        if np.abs(current_state[0]) > 25 or np.abs(current_state[0]) < 20:
            reward -= 5
            
        if np.abs(current_state[1]) < 80 and np.abs(current_state[1]) > 60:
            reward += 5
        
        if np.abs(current_state[1]) > 80 or np.abs(current_state[1]) < 60:
            reward -= 5
            
        if np.abs(current_state[2]) < 80 and np.abs(current_state[2]) > 50:
            reward += 5
        
        if np.abs(current_state[2]) > 80 or np.abs(current_state[2]) < 50:
            reward -= 5
            
        #try to encourage the agent to stay in a state whith high LAI
        if current_state[3] > 1.5 and current_state[3] < 2:
            reward -= 10
        if current_state[3] > 2 and current_state[3] < 2.5:
            reward -= 6
        if current_state[3] > 2.5 and current_state[3] < 3:
            reward -= 3
        if current_state[3] > 3 and current_state[3] < 4:
            reward -=1
        if current_state[3] > 4 :
            reward += 5
        
        # Check if temperature, humidity, or light has changed
        if (current_state[0] != previous_state[0] or
            current_state[1] != previous_state[1] or
            current_state[2] != previous_state[2]):
            reward -= 0  # Negative reward for changing temperature, humidity, or light
        else:
            reward += 0
        return reward
    
    def calculate_reward_LAI(self, previous_state, current_state):
        
        
        
        return 0

    def hort_syst(self, state, action):
        
        # Update state based on the chosen action
        #temperature_action, humidity_action, light_action = self.get_action_from_index(action)
        self.temperature_action(action)
        self.humidity_action(action)
        self.light_action(action)
        
        temperature = state[0]
        light = state[1]
        #relative_humidity = state[2]
        
        TT = self.thermal_time(temperature)
        #hourly_evapotranspiration = self.ETc(light, temperature, relative_humidity, self.hour)
        
        if self.hour % 24 ==0:
            Delta_PTI = self.delta_PTI(light, TT)
            self.PTI += Delta_PTI
            TT = []
            self.state[3] = self.Leaf_area_index()
            Delta_DMP = self.dmp_change(light)
            self.DMP += Delta_DMP
            Delta_Nup = self.nup_change(light)
            self.N_uptake += Delta_Nup
            #eTc = ET_c(HETc)
        
        #self.state[3] = LAI

    
    def get_action_from_index(self, action_index):
        """
        Convert action index to the corresponding action.
    
        Parameters:
            action_index (int): Index representing the action.
    
        Returns:
            tuple: Action representing changes in temperature, humidity, and light.
        """
        #print("Action index:", action_index)
        temperature_action = action_index // 9
        humidity_action = (action_index % 9) // 3
        light_action = action_index % 3
        return temperature_action, humidity_action, light_action
    
    def temperature_action(self, action):
        if action[0] == 0:
            self.state[0] += 0
        if action[0] == 1 and self.state[0] < 50:
            self.state[0] += 3
        if action[0] == 2 and self.state[0] > 10:
            self.state[0] -= 3
            
    def light_action(self, action):
        if action[1] == 0:
            self.state[1] += 0
        if action[1] == 1 and self.state[1] < 100:
            self.state[1] += 10
        if action[1] == 2 and self.state[1] > 0:
            self.state[1] -= 10
            
    def humidity_action(self, action):
        if action[2] == 0:
            self.state[2] += 0
        if action[2] == 1 and self.state[2] < 100:
            self.state[2] += 10
        if action[2] == 2 and self.state[2] > 0:
            self.state[2] -= 10
            
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
    def __init__(self, num_actions, num_states, temperature_bins, light_bins, humidity_bins, LAI_bins):
        """
        Initialize a Q-learning agent with given parameters.

        Parameters:
        num_actions (int): Number of possible actions.
        num_states (int): Number of possible states in the environment.
        """
        self.num_actions = num_actions
        self.num_states = num_states

        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions), dtype=float)

        # Q-learning parameters
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1
        
        # Define the bins for each continuous variable
        self.temperature_bins = temperature_bins
        self.light_bins = light_bins
        self.humidity_bins = humidity_bins
        self.LAI_bins = LAI_bins

    def select_action(self, state):
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
        humidity_index = self.discretize(state[2], self.humidity_bins)
        LAI_index = self.discretize(state[3], self.LAI_bins)
        day_value = int(state[4])  # Assuming state[4] represents the day directly
    
        # Calculate index using the provided bins
        x = temperature_index * (len(self.light_bins) * len(self.humidity_bins) * len(self.LAI_bins) * 60) + \
            light_index * (len(self.humidity_bins) * len(self.LAI_bins) * 60) + \
            humidity_index * (len(self.LAI_bins) * 60) + \
            LAI_index * 60 + \
            day_value - 1  # Subtract 1 to adjust for 0-based indexing in Python
    
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
        return len(bins) - 2


temperature_bins = [10, 15, 20, 25, 30, 35, 40]  # Define bins for temperature
light_bins = [0, 10, 20, 30, 40, 50]  # Define bins for light
humidity_bins = [0, 20, 40, 60, 80, 100]  # Define bins for humidity
LAI_bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]  # Define bins for LAI

num_actions = 27
num_states = 2200*60   #To be calculated exactly

agent = QLearningAgent(num_actions, num_states, temperature_bins, light_bins, humidity_bins, LAI_bins)


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
        
        Leaf_area_index_training.append(env.state[3])
            

        cumulative_rewards.append(total_reward)
        # Print total reward for the episode
        print(f"Episode {episode + 1}, Total LAI: {env.state[3]}")
        
        # Print episode number every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")


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

        
        Leaf_area_index_training.append(env.state[3])
            

        cumulative_rewards.append(total_reward)
        # Print total reward for the episode
        print(f"Episode {episode + 1}, Total LAI: {env.state[3]}")
        
        # Print episode number every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")

def training_loop_3():
    
    epsilon_decrement_episodes = 100  # Number of episodes after which to decrement epsilon
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
                final_LAI = state[3]
                episode_states_actions.append((next_state, None))  # None indicates the end of the episode
                break
        
        reward = -1/final_LAI
        # Update Q-table based on the episode's sequence of states and actions
        for i in range(len(episode_states_actions) - 1):
            state, action = episode_states_actions[i]
            next_state, _ = episode_states_actions[i + 1]
            #reward = (state[3]/10)**2
            
            agent.update_q_table(state, action, reward, next_state)

        
        Leaf_area_index_training.append(env.state[3])
            

        cumulative_rewards.append(reward)
        # Print total reward for the episode
        print(f"Episode {episode + 1}, Total LAI: {env.state[3]}")
        
        # Decrease epsilon every x number of episodes until it reaches min_epsilon
        if (episode + 1) % epsilon_decrement_episodes == 0 and agent.epsilon > min_epsilon:
            agent.epsilon = max(agent.epsilon - epsilon_decrement_value, min_epsilon)
            print(f"Epsilon decremented to {agent.epsilon}")
        
        # Print episode number every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")

# Get user input for version of the training loop
loop_version = input("Enter the version of the training loop to use (1 for loop_1, 2 for loop_2): ")
# Convert input to integer
loop_version = int(loop_version)
# Validate loop version input
while loop_version not in [1, 2, 3]:
    print("Invalid input. Please enter 1 for loop_1 or 2 for loop_2 or 3 for loop_3.")
    loop_version = input("Enter the version of the training loop to use (1 for loop_1, 2 for loop_2, 3 for loop_3): ")
    loop_version = int(loop_version)

if loop_version == 1:
    training_loop_1()
if loop_version == 2:
    training_loop_2()
if loop_version == 3:
    training_loop_3()
    
q_table = agent.q_table

# Calculate the mean cumulative reward at each episode
mean_cumulative_rewards = np.zeros(num_episodes)
for i in range(num_episodes):
    mean_cumulative_rewards[i] = np.mean(cumulative_rewards[:i+1])

# Calculate the mean LAI reward at each episode
mean_LAI_rewards = np.zeros(num_episodes)
for i in range(num_episodes):
    mean_LAI_rewards[i] = np.mean(Leaf_area_index_training[:i+1])

# Plot the cumulative rewards and the mean cumulative rewards
plt.figure()
plt.plot(range(1, num_episodes + 1), cumulative_rewards, label='Cumulative Reward')
plt.plot(range(1, num_episodes + 1), mean_cumulative_rewards, color='r', linestyle='--', label='Mean Cumulative Reward')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.show()

# Plot the LAI over episodes and the mean LAI reward
plt.figure()
plt.plot(range(1, num_episodes + 1), Leaf_area_index_training, label='Leaf Area Index')
plt.plot(range(1, num_episodes + 1), mean_LAI_rewards, color='r', linestyle='--', label='Mean LAI Reward')
plt.xlabel('Episodes')
plt.ylabel('Leaf Area Index')
plt.legend()
plt.show()

# Calculate moving average for cumulative rewards
window_size = 20
smoothed_rewards = np.convolve(cumulative_rewards, np.ones(window_size)/window_size, mode='valid')

# Plot the smoothed cumulative rewards
plt.figure()
plt.plot(range(window_size, num_episodes + 1), smoothed_rewards, label='Smoothed Cumulative Reward')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.show()

# Calculate moving average for LAI
smoothed_LAI = np.convolve(Leaf_area_index_training, np.ones(window_size)/window_size, mode='valid')

# Plot the smoothed LAI over episodes
plt.figure()
plt.plot(range(window_size, num_episodes + 1), smoothed_LAI, label='Smoothed Leaf Area Index')
plt.xlabel('Episodes')
plt.ylabel('Leaf Area Index')
plt.legend()
plt.show()

# Arrays to store parameter values for plotting
temperature_values = []
light_values = []
humidity_values = []
LAI_values = []


# Test agent with only max values of Q-table
agent.epsilon = 0
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
    humidity_values.append(state[2])
    LAI_values.append(state[3])

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

# Plot humidity
plt.figure(figsize=(10, 6))
plt.plot(humidity_values, label='Humidity')
plt.xlabel('Time (hours)')
plt.ylabel('Humidity')
plt.title('Relative Humidity Variation')
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