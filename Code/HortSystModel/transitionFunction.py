import numpy as np

# Define the state and action spaces
num_states = 6  # Number of environmental parameters
num_actions = 5  # Number of actions (setting light, temperature, humidity, pH, and electrical conductivity)

# Define the action space ranges and granularity
action_ranges = [
    (0, 100),  # Light intensity (arbitrary units)
    (20, 30),   # Temperature (°C)
    (40, 60),   # Relative humidity (%)
    (5.5, 6.5), # pH
    (0.5, 1.5)  # Electrical conductivity (mS/cm)
]

action_granularity = [10, 1, 5, 0.1, 0.1]

# Define the state space ranges
state_ranges = [
    (5, 9),    # pH
    (0, 3),    # Electrical conductivity (mS/cm)
    (20, 30),  # Temperature (°C)
    (40, 80),  # Relative humidity (%)
    (0, 100),  # Light intensity (arbitrary units)
    (0, 5)     # Leaf area index
]

# Initialize state variables
DMP = 2  # Initial Dry Matter Production
PTI = 10  # Initial Plant Temperature Index
N_uptake = 0.1
LAI = 1
HETc = []

# Parameters
p = {
    'Tmax': 35.00, 'Tmin': 10.00, 'Tob': 17.00, 'Tou': 24.00, 'RUE': 4.01,
    'k': 0.70, 'PTIinit': 0.025, 'a': 7.55, 'b': -0.15, 'c1': 2.82, 'c2': 74.66,
    'A': 0.3, 'Bd': 18.7, 'Bn': 8.5, 'RHmin': 0.6259, 'RHmax': 0.9398,
    'Light_day': 3.99, 'Light_night': 0.88,
}

# Accessing parameter values
Tmax, Tmin, Tob, Tou, RUE, k, PTIinit, a, b, c1, c2, A, Bd, Bn = (
    p['Tmax'], p['Tmin'], p['Tob'], p['Tou'], p['RUE'], p['k'], p['PTIinit'],
    p['a'], p['b'], p['c1'], p['c2'], p['A'], p['Bd'], p['Bn']
)

d = 3.5  # Plant crop density [m^-2]

# Functions
def PAR(R_g):
    return 0.5 * R_g

def Leaf_area_index():
    return c1 * PTI * d / (c2 + PTI)

def thermal_time(Ta):
    tt = 0
    if Ta < Tmin:
        tt = 0
    elif Tmin <= Ta < Tob:
        tt = (Ta - Tmin) / (Tob - Tmin)
    elif Tob <= Ta <= Tou:
        tt = 1
    elif Tou < Ta <= Tmax:
        tt = (Tmax - Ta) / (Tmax - Tou)
    elif Ta > Tmax:
        tt = 0
    return tt

def fi_par():
    return 1 - np.exp(-k * Leaf_area_index())

def vapor_deficit_pressure(T, RH):
    SVP = 610.78 * np.exp((T / (T + 237.3)) * 17.2694)
    return SVP * (1 - RH / 100)

def ETc(rg, T, RH, hour):
    vpd = vapor_deficit_pressure(T, RH)
    B = aerodynamic_term(hour)
    return A * (1 - np.exp(-k * Leaf_area_index())) * rg + Leaf_area_index() * vpd * B

def dmp_change(R_g):
    return RUE * fi_par() * PAR(R_g)

def ET_c(array):
    return sum(array)

def percent_n(R_g):
    DMP = dmp_change(R_g)
    return a * (DMP ** -b)

def nup_change(R_g):
    Percent_n = percent_n(R_g)
    DMP = dmp_change(R_g)
    return (Percent_n / 100) * DMP

def delta_PTI(rg, TT):
    x = TT / 24 * PAR(rg) * fi_par()
    return x

def aerodynamic_term(hour):
    return Bd if 8 < hour % 24 < 20 else Bn


def hort_syst(state, action, hour):
    
    global DMP, PTI, N_uptake, HETc, LAI
    
    light = action[0]
    temperature = action[1]
    relative_humidity = action[2]
    ph = action[3]
    ec = action[4]
    
    TT = thermal_time(temperature)
    #hourly_evapotranspiration = ETc(light, temperature, relative_humidity, hour)
    
    if hour % 24 ==0:
        Delta_PTI = delta_PTI(light, TT)
        PTI += Delta_PTI
        TT = []
        LAI = Leaf_area_index()
        Delta_DMP = dmp_change(light)
        DMP += Delta_DMP
        Delta_Nup = nup_change(light)
        N_uptake += Delta_Nup
        #eTc = ET_c(HETc)
        HETc = []
    
    
    new_state = [ph, ec, temperature, relative_humidity, light, LAI]

    # Update state variables
    state_variables = [PTI, DMP, N_uptake, HETc]
    return new_state, state_variables



def choose_action():
    """
    Choose an action by setting each of its components within the specified range.

    Returns:
        list: Chosen action with components set within the specified range.
    """
    chosen_action = []

    for i in range(len(action_ranges)):
        # Generate a random value within the specified range
        value = np.random.uniform(action_ranges[i][0], action_ranges[i][1])

        # Adjust the value to match the specified granularity
        value = round(value / action_granularity[i]) * action_granularity[i]

        # Append the adjusted value to the chosen action
        chosen_action.append(value)

    return chosen_action

# Define the simulation environment
def simulate_environment(current_state, simulation_duration):
    # Simulate the environment (replace this with your actual simulation logic)
    # For simplicity, let's randomly generate the next state and reward
    
    duration_hours = 24 * simulation_duration
    next_state = current_state  # Initialize next_state with the current state
    
    for i in range(1, duration_hours + 1):  # Start the loop from 1
        chosen_action = choose_action()  # Choose a new action for each iteration
        next_state = hort_syst(next_state, chosen_action, i)
    
    reward = np.random.uniform(0, 1)
    return next_state, reward


# Example usage of the environment
initial_state = [7, 1, 25, 60, 50, 1]  # Initial state (replace with actual initial state)
simulation_duration = 60

next_state, reward = simulate_environment(initial_state, simulation_duration)

# Print the results
print("Simulation Results:")
print(f"{'='*20}\n")
print("Initial State:")
for i, (label, value) in enumerate(zip(['pH', 'EC', 'Temperature', 'Humidity', 'Light Intensity', 'Leaf Area Index'], initial_state)):
    print(f"{label}: {value}{' (within range)' if state_ranges[i][0] <= value <= state_ranges[i][1] else ' (outside range)'}")

print(f"\nChosen Action:")
#for i, (label, value) in enumerate(zip(['Light Intensity', 'Temperature', 'Relative Humidity', 'pH', 'Electrical Conductivity'], chosen_action)):
    #print(f"{label}: {value}{' (within range)' if action_ranges[i][0] <= value <= action_ranges[i][1] else ' (outside range)'}")

print(f"\nFinal State:")
for i, (label, value) in enumerate(zip(['pH', 'EC', 'Temperature', 'Humidity', 'Light Intensity', 'Leaf Area Index'], next_state[0])):
    print(f"{label}: {value}{' (within range)' if state_ranges[i][0] <= value <= state_ranges[i][1] else ' (outside range)'}")

print(f"\nState Variables:")
state_variable_labels = ['PTI', 'DMP', 'N Uptake', 'HETc']
for label, value in zip(state_variable_labels, next_state[1]):
    print(f"{label}: {value}")

print(f"\nReward: {reward}\n")
print(f"{'='*20}")