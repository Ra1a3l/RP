import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from HortSystEnvironmentSimplified import HortSystSimplified
from QLearningAgent import QLearningAgent
from DQNAgent import DeepQNAgent

# Define the number of actions and states
num_actions = 9
num_states = 60000   # Only for the Q_table

# Define the exploration strategies
epsilon_based_training = 'epsilon-greedy'
boltzmann_based_training = 'boltzmann'

# Initialize Deep Q-N Agent with the state shape (4,) and action shape 9
agent_DQN = DeepQNAgent((4,), num_actions)
# Initialize Q-learning Agent with the number of actions, states, and exploration strategy
agent_QL = QLearningAgent(num_actions, num_states, epsilon_based_training, temperature=1.0)
# Initialize the environment with the Deep Q-N Agent
env = HortSystSimplified()

# Print the number of available GPUs
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def training_Q_learning(exploration_strategy):
    num_episodes = 100000
    agent_QL.epsilon = 0.2
    agent_QL.exploration_method = exploration_strategy
    cumulative_rewards = []
    Leaf_area_index_training = []
    q_table_filled_percentage = []  # List to store percentage of Q-table filled
    exploration_rates = []  # List to store exploration rates
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        
        state = env.state
        
        while True:
            action = agent_QL.select_action(state)

            next_state, reward, done, _ = env.step(action)

            agent_QL.update_q_table(env.state, action, reward, next_state)

            total_reward += reward
            state = next_state
            
            if done:
                break
        
        Leaf_area_index_training.append(env.state[2])
            
        cumulative_rewards.append(total_reward)
        
        # Calculate percentage of Q-table filled every 10 episodes
        if (episode + 1) % 10 == 0:
            filled_percentage = agent_QL.percentage_filled()
            q_table_filled_percentage.append(filled_percentage)
        
        # Record the exploration rate
        exploration_rates.append(agent_QL.epsilon)
        
        # Print episode number every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")

    # Plotting all metrics in subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Cumulative rewards plot
    axs[0, 0].plot(cumulative_rewards)
    axs[0, 0].set_title('Cumulative Rewards over Episodes')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Cumulative Reward')

    # Last value of LAI plot
    axs[0, 1].plot(Leaf_area_index_training)
    axs[0, 1].set_title('Last Value of LAI over Episodes')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Last Value of LAI')

    # Percentage of Q-table filled plot
    axs[1, 0].plot(q_table_filled_percentage)
    axs[1, 0].set_title('Percentage of Q-table Filled over Episodes')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Percentage of Q-table Filled')

    # Exploration rate plot
    axs[1, 1].plot(exploration_rates)
    axs[1, 1].set_title('Exploration Rate over Episodes')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Exploration Rate')

    plt.tight_layout()
    plt.show()

def training_DQN():
    num_episodes = 10
    final_lai_values = []
    cumulative_rewards = []
    exploration_rates = []  # List to store exploration rates
    epsilon = 1.0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent_DQN.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            if done:
                final_lai = next_state[2]
                reward = env.calculate_reward(state)
                agent_DQN.remember(state, action, reward, next_state, done)
            agent_DQN.train()
            if len(agent_DQN.replay_memory) % 10 == 0:
                agent_DQN.update_target_model()
            total_reward += reward
            state = next_state
        final_lai_values.append(final_lai)
        cumulative_rewards.append(total_reward)
        exploration_rates.append(epsilon)
        if epsilon > 0.1:
            epsilon -= 0.01
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Final LAI: {final_lai}, Epsilon: {epsilon}")
        
    # Plotting all metrics in subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Cumulative rewards plot
    axs[0, 0].plot(cumulative_rewards)
    axs[0, 0].set_title('Cumulative Rewards over Episodes')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Cumulative Reward')

    # Last value of LAI plot
    axs[0, 1].plot(final_lai_values)
    axs[0, 1].set_title('Last Value of LAI over Episodes')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Last Value of LAI')

    # Exploration rate plot
    axs[1, 1].plot(exploration_rates)
    axs[1, 1].set_title('Exploration Rate over Episodes')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Exploration Rate')

    plt.tight_layout()
    plt.show()

def main():
    print("Choose the type of agent:")
    print("1. Q-learning")
    print("2. Deep Q-N")
    
    agent_choice = int(input("Enter the number of the agent type: "))
    
    if agent_choice == 1:
        print("Choose the exploration strategy:")
        print("1. Epsilon-greedy")
        print("2. Boltzmann")
        print("3. Both")
        strategy_choice = int(input("Enter the number of the strategy: "))
        
        if strategy_choice == 1:
            training_Q_learning('epsilon-greedy')
        elif strategy_choice == 2:
            training_Q_learning('boltzmann')
        elif strategy_choice == 3:
            training_Q_learning('epsilon-greedy')
            training_Q_learning('boltzmann')
        else:
            print("Invalid input. Please choose a number between 1 and 3.")
    elif agent_choice == 2:
        training_DQN()
    else:
        print("Invalid input. Please choose either 'Q-learning' or 'Deep Q-N'.")

if __name__ == "__main__":
    main()
