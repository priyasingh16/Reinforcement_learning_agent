import tensorflow as tf
import numpy as np
from collections import deque
from random import sample


class DeepQModel:
    """
        Description
        -----------
            the class deals with building the right parameters and functions to
            built a deep learning agent that learn and play games
        Parameters
        ----------
        input_shape : Integer
            the shape for the input states
        output_shape : Integer
            the number of possible actions
        learning_rate : Double
            learning rate for the optimizer for neural network
        gamma : Double
            discount factor to be multiplied with the future rewards
    """

    def __init__(self, input_shape, output_shape, learning_rate, gamma):
        """
        Description
        -----------
        Initialize shapes, learning_rate, decay and epsilon
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.replayBuffer = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.learning_rate = 0.001
        self.predictionModel = self.buildmodel()
        self.targetModel = self.buildmodel()
        self.targetModel.set_weights(self.predictionModel.get_weights())

    def syncNetworks(self):
        self.targetModel.set_weights(self.predictionModel.get_weights())

    def predict(self, state):
        return self.predictionModel.predict(state)

    def buildmodel(self):
        """
        Description
        -----------
            The function here takes no inputs and outputs a keras model.
            keras deep learning model which takes state as input and outputs
            all the Q values for all possible actions.
            Its a deep neural network with 2 hidden layers and one output layer
            with  relu activation for hidden layer and linear activations for
            output action Q value pair layer.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, 8, strides=(4, 4), padding='valid', input_shape=self.input_shape,
                                         activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, 4, strides=(2, 2), padding='valid', activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.output_shape, activation='linear'))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        model.summary()
        return model

    def appendReplay(self, data):
        """
        Description
        -----------
        Keep a set of 2000 most recend record which will be used for training
        so the neural network doesnt forget the Q value estimates
        Parameters
        ----------
        data : tuple
            contains state, action, reward, next_state, done as a tuple
        """
        self.replayBuffer.append(data)

    def decay(self):
        if self.epsilon > 0.01:
            self.epsilon = 0.995 * self.epsilon
        else:
            self.epsilon = 0.01

    def epsilon_condition(self):
        return np.random.rand() <= self.epsilon

    def load_model(self, path):
        model = tf.keras.models.load_model(path)
        return model

    def train(self, batchSize=32):
        """
        Parameters
        ----------
        model : keras model
            A Sequential model from keras to predict the stage action rewards
        batchSize : Integer
            The batch to be trained after every play
        """
        if batchSize < len(self.replayBuffer):
            samples = sample(self.replayBuffer, batchSize)
        else:
            samples = self.replayBuffer
        for observation in samples:
            state, action, reward, next_state, done = observation
            if done is True:
                t = reward
            else:
                next_state_max_reward = np.amax(self.targetModel.predict(next_state))
                t = reward + (self.gamma * next_state_max_reward)

            target_action_pair = self.predictionModel.predict(state)
            target_action_pair[0][action] = t
            self.predictionModel.fit(state, target_action_pair, epochs=1, verbose=0)
        self.decay()







