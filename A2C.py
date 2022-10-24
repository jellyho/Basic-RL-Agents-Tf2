import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

class NN(tf.keras.Model):
  def __init__(self, action_size):
    super(NN, self).__init__()
    self.actor_fc = Dense(24, activation='tanh')
    self.actor_out = Dense(action_size, activation='softmax', kernel_initializer=RandomUniform(-1e-3, 1e-3))

    self.critic_fc1 = Dense(24, activation='tanh')
    self.critic_fc2 = Dense(24, activation='tanh')
    self.critic_out = Dense(1, kernel_initializer=RandomUniform(-1e-3, 1e-3))

  def __call__(self, x):
    actor_x = self.actor_fc(x)
    policy = self.actor_out(actor_x)

    critic_x = self.critic_fc1(x)
    critic_x = self.critic_fc2(critic_x)
    value = self.critic_out(critic_x)

    return policy, value
  
class A2C:
  def __init__(self, action_size):
    self.render = False

    self.action_size = action_size

    self.discount_factor = 0.99
    self.learning_rate = 0.001

    self.model = NN(self.action_size)
    self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=5.0)
  
  def get_action(self, state):
    policy, _ = self.model(state)
    policy = np.array(policy[0])
    return np.random.choice(self.action_size, 1, p=policy)[0]

  def train_model(self, state, action, reward, next_state, done):
    model_params = self.model.trainable_variables

    with tf.GradientTape() as tape:
      policy, value = self.model(state)
      _, next_value = self.model(next_state)
      target = reward + (1 - done) * self.discount_factor * next_value[0]

      one_hot = tf.one_hot([action], self.action_size)
      action_prob = tf.reduce_sum(one_hot * policy, axis=1)
      adv = tf.stop_gradient(target - value[0])
      actor_loss = tf.math.log(action_prob + 1e-5) * adv
      actor_loss = -tf.reduce_mean(actor_loss)

      critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
      critic_loss = tf.reduce_mean(critic_loss)

      loss = 0.2 * actor_loss + critic_loss
    
    gradient = tape.gradient(loss, model_params)
    self.optimizer.apply_gradients(zip(gradient, model_params))

    return np.array(loss)
