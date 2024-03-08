import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

class DeepQNAgent:
    def __init__(self, state_shape, action_shape, learning_rate=0.0001, discount_factor=0.95):
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
        self.min_replay_size = 600
        self.batch_size = 60

    def build_model(self):
        """Builds and returns the deep Q-network model."""
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_shape=self.state_shape, activation='relu', kernel_initializer='he_uniform'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.action_shape, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
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