import numpy as np

parameters = {
    'Tmax': 35.00,         # Maximum temperature (°C)
    'Tmin': 10.00,         # Minimum temperature (°C)
    'Tob': 17.00,          # Lower optimal temperature (°C)
    'Tou': 24.00,          # Upper optimal temperature (°C)
    'RUE': 4.01,           # Radiation use efficiency (g MJ^-1)
    'k': 0.70,             # Extinction coefficient
    'PTIini': 0.025,       # PTI initial condition (MJ d^-1)
    'N_conc': 7.55,        # N concentration in dry biomass at the end of the exponential growth period (g m^-2)
    'b': -0.15,            # Slope of the relationship
    'c1': 2.82,            # Slope of the curve (m^-2)
    'c2': 74.66,           # Intersection coefficient
    'A': 0.3,              # Radiative coefficient
    'Bd': 18.7,            # Daytime aerodynamic coefficient during
    'Bn': 8.5,             # Nighttime serodynamic coefficient
    'RHmin': 0.6259,
    'RHmax': 0.9398,
    'Light_day': 3.99,
    'Light_night': 0.88,
}

# Accessing the parameter values
Tmax = parameters['Tmax']
Tmin = parameters['Tmin']
Tob = parameters['Tob']
Tou = parameters['Tou']
RUE = parameters['RUE']
k = parameters['k']
PTIinit = parameters['PTIini']
a = parameters['N_conc']
b = parameters['b']
c1 = parameters['c1']
c2 = parameters['c2']
A = parameters['A']
Bd = parameters['Bd']
Bn = parameters['Bn']
d = 3.5       #Plant crop density [m^-2]


def PAR(R_g):
    """
    Calculate photosynthetically active radiation (PAR) as half of the solar radiation.

    Args:
        R_g (float): Solar radiation.

    Returns:
        float: PAR value.
    """
    return 0.5 * R_g


def Leaf_area_index():
    """
    Calculate leaf area index (LAI) based on c1, PTI, d, and c2.

    Returns:
        float: LAI value.
    """
    return c1 * PTI * d / (c2 + PTI)


def thermal_time(Ta):
    """
    Calculate thermal time based on temperature (Ta) and specified temperature thresholds.

    Args:
        Ta (float): Ambient temperature.

    Returns:
        float: Thermal time value.
    """
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
    """
    Calculate fi-PAR based on leaf area index (LAI) and coefficient k.

    Args:
        LAI (float): Leaf area index.

    Returns:
        float: fi-PAR value.
    """
    
    LAI = Leaf_area_index()
    
    return 1 - np.exp(-k * LAI)


def vapor_deficit_pressure(T, RH):
    """
    Calculate the vapor deficit pressure.

    Args:
        T (float): Temperature in Celsius.
        RH (float): Relative humidity in percentage.

    Returns:
        float: Vapor deficit pressure in Pascal.
    """
    SVP = 610.78 * np.exp((T / (T + 237.3)) * 17.2694)
    print(SVP * (1 - RH / 100))
    return SVP * (1 - RH / 100)

def ETc(rg, T, RH, hour):
    """
    Calculate hourly ETc (crop evapotranspiration) based on LAI, solar radiation, temperature, relative humidity,
    and coefficients A, B.

    Args:
        
        rg (float): Solar radiation.
        T (float): Temperature in Celsius.
        RH (float): Relative humidity in percentage.
        hour(float): hour since start of simunlation.

    Returns:
        float: ETc value.
    """
    vpd = vapor_deficit_pressure(T, RH)
    B = aerodynamic_term(hour)
    LAI = Leaf_area_index()
    
    return A * (1 - np.exp(-k * LAI)) * rg + LAI * vpd * B


def dmp_change(R_g):
    """
    Calculate the change in dry matter production (ΔDMP) based on fi-PAR and PAR.

    Args:
        fi_par (float): fi-PAR value.
        par (float): Photosynthetically active radiation (PAR).

    Returns:
        float: ΔDMP value.
    """
    
    fipar = fi_par()
    par = PAR(R_g)
    return RUE * fipar * par


def ET_c(array):
    """
    Calculate average daily ETc (crop evapotranspiration) over 24 hours based on LAI, solar radiation (rg),
    and vapor pressure deficit (vpd).

    Args:
        LAI (float): Leaf area index.
        rg (float): Solar radiation.
        vpd (float): Vapor pressure deficit.

    Returns:
        float: daily ETc value.
    """
    x = 0
    
    for i in range(len(array)):
        x += array[i]

    return x


def percent_n(R_g):
    """
    Calculate percentage change in nitrogen (%ΔN) based on change in dry matter production (ΔDMP)
    and parameters a and b.

    Args:
        dmp (float): Change in dry matter production.
        a (float): Coefficient a.
        b (float): Coefficient b.

    Returns:
        float: %ΔN value.
    """
    
    DMP = dmp_change(R_g)
    
    return a * (DMP ** -b)


def nup_change(R_g):
    """
    Calculate change in nitrogen uptake (ΔNup) based on percentage change in nitrogen (%ΔN) and ΔDMP.

    Args:
        percent_n (float): Percentage change in nitrogen.
        dmp (float): Change in dry matter production.

    Returns:
        float: ΔNup value.
    """
    Percent_n = percent_n(R_g)
    DMP = dmp_change(R_g)
    return (Percent_n / 100) * DMP



def delta_PTI(rg, TT):
    """
    Calculate the change in PTI (Plant Temperature Index) based on ambient temperature (TA), solar radiation (rg),
    and leaf area index (LAI).

    Args:
        TA (float): Ambient temperature.
        rg (float): Solar radiation.
        LAI (float): Leaf area index.

    Returns:
        float: Change in PTI value.
    """
    x = 0
    for i in range(len(TT)):
        x += TT[i]
    x = x / 24
    x = x * PAR(rg) * fi_par()
    return x

def aerodynamic_term(hour):
    
    if (hour % 24 > 8 and hour % 24 < 20):
        return Bd
    else:
        return Bn
    

# Simulation parameters
simulation_duration = 60  # Number of days to simulate
hours_per_day = 24  # Number of hours per day

import random

def generate_temperature(hour):
    """
    Generate a random temperature value for the given hour.

    Args:
        hour (int): Hour of the day (0-23).

    Returns:
        float: Random temperature value within the specified bounds.
    """
    
    # Generate a random temperature value within the bounds
    temperature = random.uniform(18, 25)
    return temperature

def generate_humidity(hour):
    """
    Generate a random humidity value for the given hour.

    Args:
        hour (int): Hour of the day (0-23).

    Returns:
        float: Random humidity value within the specified bounds.
    """
    RHmin = parameters['RHmin']
    RHmax = parameters['RHmax']
    # Generate a random humidity value within the bounds
    humidity = random.uniform(RHmin, RHmax)
    return humidity

def generate_light(hour):
    """
    Generate the light value (Rg) for the given hour.

    Args:
        hour (int): Hour of the day (0-23).

    Returns:
        float: Constant light value based on the hour (day or night).
    """
    if hour >= 6 and hour < 18:
        # Daytime: Set the constant light value for the day
        light = parameters['Light_day']
    else:
        # Nighttime: Set the constant light value for the night
        light = parameters['Light_night']
    return light

def generate_inputs():
    """
    Generate inputs for each hour of the 60-day simulation.

    Returns:
        tuple: Arrays of temperature, humidity, and light values.
    """
    temperature_values = []
    humidity_values = []
    light_values = []

    for day in range(1000):
        for hour in range(24):
            temperature = generate_temperature(hour)
            humidity = generate_humidity(hour)
            light = generate_light(hour)

            temperature_values.append(temperature)
            humidity_values.append(humidity)
            light_values.append(light)

    return temperature_values, humidity_values, light_values



# Simulation data storage
input_measurements = []  # Store input measurements
output_results = []  # Store model outputs

#Simulation initialization

temperature_input, humidity_input, light_input = generate_inputs()

# Access the temperature, humidity, and light values
Temperature_input = temperature_input
Humidity_input = humidity_input
Light_input = light_input

# Initialize variables
hours = 0  # Total hours elapsed since start of the experiment
day = 1  # Day counter
LAI = 5  # Initial Leaf Area Index
PTI = 10  # Initial Plant Temperature Index
DMP = 2  # Initial Dry Matter Production
N_uptake = 0.1
eTc = 0.1

TT = []
HETc = []

# Arrays to store output values
LAI_values = []
PTI_values = []
DMP_values = []
N_uptake_values = []
Daily_Evapotranspiration_values = []
days = []  # Array to store day values

# Main simulation loop
while hours < 60 * 24:  # Run the simulation for 60 days (60 * 24 hours)
    # Update input variables (temperature, humidity, light, etc.)
    
    temperature = Temperature_input[hours]
    humidity = Humidity_input[hours]
    light = Light_input[hours]


    # Calculate TT, PAR, ETC, and other variables for the current hour
    TT.append(thermal_time(temperature))
    Hourly_evapotranspiration = ETc(light, temperature, humidity, hours)
    HETc.append(Hourly_evapotranspiration)


    # Check if 24 hours have passed (end of a day)
    if hours % 24 == 0:
        # Update variables for a new day
        day += 1
        
        Delta_PTI = delta_PTI(light, TT)
        PTI += Delta_PTI
        TT = []
        LAI = Leaf_area_index()
        Delta_DMP = dmp_change(light)
        DMP += Delta_DMP
        Delta_Nup = nup_change(light)
        N_uptake += Delta_Nup
        eTc = ET_c(HETc)
        HETc = []
        
        # Store output values in respective arrays
        LAI_values.append(LAI)
        PTI_values.append(PTI)
        DMP_values.append(DMP)
        N_uptake_values.append(N_uptake)
        Daily_Evapotranspiration_values.append(eTc)
        days.append(day)
        
        
        # Reset or update any variables that need to be reset or updated daily

    # Increment the hour counterso
    hours += 1

# Simulation is complete

import matplotlib.pyplot as plt

# Plotting the graphs
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 12))
fig.tight_layout(pad=3.0)  # Adjust the spacing between subplots

# LAI plot
axes[0].plot(days, LAI_values, 'b-')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('LAI [m2/m2]')

# PTI plot
axes[1].plot(days, PTI_values, 'g-')
axes[1].set_xlabel('Day')
axes[1].set_ylabel('PTI [MJ/m2]')

# DMP plot
axes[2].plot(days, DMP_values, 'r-')
axes[2].set_xlabel('Day')
axes[2].set_ylabel('DMP [g/m2]')

# N_uptake plot
axes[3].plot(days, N_uptake_values, 'm-')
axes[3].set_xlabel('Day')
axes[3].set_ylabel('N_uptake [g/m2]')

# Daily Evapotranspiration plot
axes[4].plot(days, Daily_Evapotranspiration_values, 'c-')
axes[4].set_xlabel('Day')
axes[4].set_ylabel('Evapotranspiration [g/m2]')

plt.show()
