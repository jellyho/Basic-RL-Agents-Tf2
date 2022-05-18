import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import defaultdict

class NN(tf.keras.Model):
  def __init__(self, action_size):
    super(NN, self).__init__()
    self.fc1 = Dense(30, activation='relu')
    self.fc2 = Dense(60, activation='relu')
    self.fc_out = Dense(action_size)

  def call(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    q = self.fc_out(x)
    return q

class DEEP_SARSA:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    
    self.discount = 0.99
    self.epsilon = 1
    self.learning_rate = 0.001
    self.epsilon_decay = 0.999
    self.epsilon_min = 0.01

    self.model = NN(self.action_size)
    self.optimizer = Adam(learning_rate=self.learning_rate)

  def get_action(self, state):
    if np.random.rand() < self.epsilon:
      return random.randrange(self.action_size)
    else:
      q = self.model(state)[0]
      return np.argmax(q)

  def learn(self, state, action, reward, next_state, next_action, done):
    if self.epsilon > self.epsilon_min:
      self.epsilon = self.epsilon * self.epsilon_decay
    
    model_params = self.model.trainable_variables

    with tf.GradientTape() as tape:
      tape.watch(model_params)
      predict = self.model(state)[0]
      predict = tf.reduce_sum(tf.one_hot([action], self.action_size) * predict, axis=1)

      next_q = self.model(next_state)[0][next_action]

      target = reward + (1-done) * self.discount * next_q

      loss = tf.reduce_mean(tf.square(target - predict))

    grads = tape.gradient(loss, model_params)
    self.optimizer.apply_gradients(zip(grads, model_params))
