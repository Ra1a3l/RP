import numpy as np

class HortSystSimplified:
    def __init__(self):
        self.state_space = 4
        self.state = np.zeros(4)
        self.action_space = np.arange(9)  # 9 different actions
        self.simulation_time = 60*24
        self.num_days = 60
        
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
        
        #Define target LAI and tolerance
        self.target = 4  # Example target LAI
        self.tolerance = 0.2  # Example tolerance value
        
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
        
        self.hort_syst(self.state, action)  # Update the environment for each day
        
        # Calculate reward based on the changes during the day
        reward = self.calculate_reward(self.state)
        
        # Increment the day
        self.state[3] += 1
        
        # Reset hour to 0 for the next day
        self.hour = 0
        
        # Check if simulation has ended
        if self.state[3] >= self.num_days:  # Assuming self.num_days represents the total number of days
            self.done = True 
        
        return self.state, reward, self.done, {}

    def calculate_reward(self, current_state):
        # Check if the LAI is within the target range
        if abs(current_state[2] - self.target) <= self.tolerance:
            return 1  # Sparse binary reward: +1 for achieving the goal LAI within tolerance
        else:
            return 0   # Sparse binary reward: 0 for all other states

    def generate_random_goal(self):
        """
        Generates a random goal for the agent.

        Returns:
            float: A random goal value between 2 and 4.5.
        """
        return np.random.uniform(4, 4.5)
    
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
            self.state[0] += 3
        if action[0] == 2 and self.state[0] > 13:
            self.state[0] -= 3
            
    def light_action(self, action):
        if action[1] == 0:
            self.state[1] += 0
        if action[1] == 1 and self.state[1] <= 90:
            self.state[1] += 10
        if action[1] == 2 and self.state[1] >= 10:
            self.state[1] -= 10
            
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