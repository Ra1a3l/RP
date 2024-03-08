import numpy as np

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