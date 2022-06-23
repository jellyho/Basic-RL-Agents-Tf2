import random
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
import tensorflow as tf

class NN(tf.keras.Model):
  def __init__(self, action_size):
    super(NN, self).__init__()
    self.fc1 = Dense(24, activation='relu')
    self.fc2 = Dense(24, activation='relu')
    self.fc_out = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3))

  def call(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    q = self.fc_out(x)
    return q
  
class DQN:
  def __init__(self, state_size, aciton_size):
    self.render = False

    self.state_size = state_size
    self.action_size = aciton_size

    self.discount_factor = 0.99
    self.learning_rate = 0.001
    self.epsilon = 1.0
    self.epsilon_decay = 0.999
    self.epsilon_min = 0.001
    self.batch_size = 64
    self.train_start = 1000

    self.memory = deque(maxlen=2000)

    self.model = NN(self.action_size)
    self.target_model = NN(self.action_size)
    self.optimizer = Adam(learning_rate=self.learning_rate)

    self.update_target_model()

  def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())

  def get_action(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    else:
      q = self.model(state) #리스트 형태로 반환됨
      return np.argmax(q[0])

  def append_sample(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def train_model(self):
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

    mini_batch = random.sample(self.memory, self.batch_size)
    states = np.array([sample[0][0] for sample in mini_batch])
    actions = np.array([sample[1] for sample in mini_batch])
    rewards = np.array([sample[2] for sample in mini_batch])
    next_states = np.array([sample[3][0] for sample in mini_batch])
    dones = np.array([sample[4] for sample in mini_batch])

    model_params = self.model.trainable_variables

    with tf.GradientTape() as tape:
      predicts = self.model(states)
      one_hot_action = tf.one_hot(actions , self.action_size)
      predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

      target_predicts = self.target_model(next_states)
      target_predicts = tf.stop_gradient(target_predicts)

      max_q = np.amax(target_predicts, axis=-1)
      targets = rewards + (1-dones) * self.discount_factor * max_q
      loss = tf.reduce_mean(tf.square(targets - predicts))

    grads = tape.gradient(loss, model_params)
    self.optimizer.apply_gradients(zip(grads, model_params))
