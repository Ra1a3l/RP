import tensorflow as tf
import numpy as np
import random
import time
import matplotlib.pyplot as plt

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
        
        return self.state, 4

    def step(self, action_index):
        action = self.get_action_from_index(action_index)
        previous_state = np.copy(self.state)  # Store the previous state
        
        self.hort_syst(self.state, action)  # Update the environment for each day
        
        # Calculate reward based on the changes during the day
        reward = self.calculate_reward(previous_state, self.state)
        
        # Store transitions with hindsight goals for HER
        achieved_goal = self.state[2]  # LAI is the goal
        self.agent.remember(previous_state, action_index, reward, self.state, self.done, achieved_goal)
        
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

class Episode_experience():
    def __init__(self):
        self.memory = []
        
    def add(self, state, action, reward, next_state, done, goal):
        self.memory += [(state, action, reward, next_state, done, goal)]
        
    def clear(self):
        self.memory = []
        
class DQNAgent():
    def __init__(self, state_size, action_size, goal_size, use_double_dqn=True, clip_target_value=True):
        self.state_size = state_size
        self.goal_size = goal_size
        self.action_size = action_size
        self.use_double_dqn = use_double_dqn
        self.clip_target_value = clip_target_value
        self.memory = []
        self.epsilon = 0.2 # exploration
        self.epsilon_min = 0.02 # min exploration
        self.epsilon_decay = 0.95
        self.tau = 0.95 # target net update weight
        self.gamma = 0.98
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.buffer_size = int(1e6)
        self._set_model()
        
    def _set_model(self): # set value network
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.state_size + self.goal_size,)),
            tf.keras.layers.Dense(self.action_size)
        ])

        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def choose_action(self, state, goal):
        state = np.array(state).flatten()  # Ensure state is 1D array
        goal = np.array(goal).flatten()  # Ensure goal is 1D array
        
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(np.hstack([state, goal]).reshape(1, -1))
        return np.argmax(act_values[0])

    
    def remember(self, ep_experience):
        self.memory += ep_experience.memory
        if len(self.memory) > self.buffer_size:
            self.memory = self.memory[-self.buffer_size:] # empty the first memories
        
    def replay(self, optimization_steps):
        if len(self.memory) < self.batch_size: # if there's no enough transitions, do nothing
            return 0
        
        losses = 0
        for _ in range(optimization_steps):
            minibatch = random.sample(self.memory, self.batch_size)
            states = np.array([np.hstack([e[0], e[5]]) for e in minibatch])
            actions = np.array([e[1] for e in minibatch])
            rewards = np.array([e[2] for e in minibatch])
            next_states = np.array([np.hstack([e[3], e[5]]) for e in minibatch])
            dones = np.array([e[4] for e in minibatch])
            goals = np.array([e[5] for e in minibatch])
            
            target = rewards + self.gamma * (1 - dones) * np.max(self.target_model.predict(next_states), axis=1)
            target_full = self.model.predict(states)
            target_full[np.arange(len(actions)), actions] = target
            
            loss = self.model.train_on_batch(states, target_full)
            losses += loss
            
        return losses/optimization_steps # return mean loss
    
    def update_target_net(self, decay=True):
        if decay:
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
            
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)
            
            

size = 50
agent = DQNAgent(4, 9, 4)
env = HorticultureEnvironment(agent)

use_her = True # use hindsight experience replay or not
num_epochs = 250
num_episodes = 16
optimization_steps = 40
K = 4 # number of random future states

losses = []
success_rate = []

ep_experience = Episode_experience()

start = time.clock()
for i in range(num_epochs):
    successes = 0
    for n in range(num_episodes):
        state, goal = env.reset()
        for t in range(size):
            action = agent.choose_action(state, goal)  # Pass state and goal directly without wrapping them in lists
            next_state, reward, done, _ = env.step(action)
            ep_experience.add(state, action, reward, next_state, done, goal)
            state = next_state
            if done:
                break
        successes += done

        if use_her: # The strategy can be changed here
            #         goal = state # HER, with substituted goal=final_state
            for t in range(len(ep_experience.memory)):
                for k in range(K):
                    future = np.random.randint(t, len(ep_experience.memory))
                    goal = ep_experience.memory[future][3] # next_state of future
                    state = ep_experience.memory[t][0]
                    action = ep_experience.memory[t][1]
                    next_state = ep_experience.memory[t][3]
                    done = np.array_equal(next_state, goal)
                    reward = 0 if done else -1
                    ep_experience.add(state, action, reward, next_state, done, goal)

        agent.remember(ep_experience)  
        ep_experience.clear()
        
        mean_loss = agent.replay(optimization_steps)
    agent.update_target_net()
    
    losses.append(mean_loss)
    success_rate.append(successes/num_episodes)
    print("\repoch", i+1, "success rate", success_rate[-1], 'loss %.2f'%losses[-1], 'exploration %.2f'%agent.epsilon, end=' '*10)

print("Training time : %.2f"%(time.clock()-start), "s")

agent.saver.save(agent.sess, 'model/'+str(size)+'bits.ckpt')

plt.title('Q loss')
plt.plot(losses)
plt.show()


