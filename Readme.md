

# Reinforcement Learning Implementation for Hydroponic System Control

## Overview
This project aims to train an AI agent to control a hydroponic system, optimizing its parameters for efficient plant growth and resource utilization. The project leverages reinforcement learning techniques to enable the agent to learn and adapt its behavior over time, aiming to maximize the Leaf Area Index (LAI) while minimizing resource consumption.

## Environment Modeling
The environment is modeled using the template from the OpenAI Gym library, which provides a flexible framework for defining and interacting with reinforcement learning environments. Key components of the environment include sensors to provide state information (such as temperature, light, humidity, LAI), and actuators to control system parameters.

## Goal Definition
The optimization goal is to maximize plant growth (represented by LAI and Dry Matter Production, DMP) while minimizing resource consumption (nutrient uptake, temperature, light, humidity). The agent's control actions are limited to adjusting temperature, light, and humidity levels within the system. The overarching objective is to find the optimal sequence of actions that maximize LAI while minimizing resource usage.

## Description of different folders
Inside the code folder (all programs are written in python), you will find the implementation of the HortSyst Model, that you can play with.

In the ReinforcementLearning folder you will find two different environments, the PlantPositionRL is a simple environment simulating the movement of a robotic arm which goal is to make measurements and come back to the initial position. This was just a training program I did to test on a simpler, smaller environment. There is also an attempt at an implementation in DeepQN which is not working due to the data type of the state. I have not continued working on it for this environment but a similar implentation is being tried for the HortSyst environment

The HortSyst environment is the main program that is used to train a q-table in the HortSyst environment. I contains a class for the environment and a class for the agent. when running the program you will be asked to enter the number of episode you wish to run the training for (test for 100 to 1000 episodes at first to see how long it runs on you machine: a few seconds to 2-3 minutes). Then you will be asked to choose the training loop you wish to use. There are currently 3 options.

- Training loop 1: Uses the reward function as defined in the environment class, it gives rewards every step without assessing the end result for LAI
- Training loop 2: Takes into account the whole sequence of action and calculates the reward at the end of the episode. It then loops back through the saved state-action pairs and attribute the rewards, updating the q-table.
- Training loop 3: Just a prototype to start with an epsilon value of 1 for exploration only of the environment, then decreases epsilon until reaching a value of 0.1. This was to test if giving more exploration time lead to better training and avoid overfitting.

There are also plots giving the evolution of the final value of LAI as a function of number of episode, the cumulative reward and averaged smoothed out plots for better visualization.

There is a last simulation at the end of the code simulating one environment with epsilon set to 0 to get the best set of actions of our q-table after the training. There is plots showing LAI as a function of time (hours) and plots for the temperature, humidity and light as a function of time as well.

Under the results folder, you will find graphs coming from simulations of the 2 environments. The titles should give how many episodes, which paramter is plotted and what reward function was used. This will be cleaned up and updated when better simulations will be available. For now it just serves as initial results to compare with and keep a trace of older reward function used.

## Dependencies

- Python
- Numpy
- Matplotlib

For DeepQN

- Tensorflow
- Keras
- Collections


