import sys
import gym
import pylab
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from tensorflow_probability import distributions as tfd

class NN(tf.keras.Model):
    def __init__(self, action_size):
        super(NN, self).__init__()
        self.actor_fc1 = Dense(24, activation='tanh')
        self.actor_mu = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3))
        self.actor_sigma = Dense(action_size, activation='sigmoid', kernel_initializer=RandomUniform(-1e-3, 1e-3))

        self.critic_fc1 = Dense(24, activation='tanh')
        self.critic_fc2 = Dense(24, activation='tanh')
        self.critic_out = Dense(1, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        actor_x = self.actor_fc1(x)
        mu = self.actor_mu(actor_x)
        sigma = self.actor_sigma(actor_x)

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)

        return mu, sigma, value

class CA2C:
    def __init__(self, action_size, max_action):
        self.render = False
        self.action_size = action_size
        self.max_action = max_action

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = NN(self.action_size)

        self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)

    def get_action(self, state):
        mu, sigma, _ = self.model(state)
        dist = tfd.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0]
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    def train_model(self, state, action, reward, next_state, done):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            mu, sigma, value = self.model(state)
            _, _, next_value = self.model(next_state)
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            adv = tf.stop_gradient(target - value[0])
            dist = tfd.Normal(loc=mu, scale=sigma)
            action_prob = dist.prob([action])[0]
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            actor_loss = tf.reduce_mean(cross_entropy * adv)

            critic_loss = 0.5 * tf.math.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            loss = 0.1 * actor_loss + critic_loss

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return loss, sigma


env = gym.make('MountainCarContinuous-v0', render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
max_action = env.action_space.high[0]

agent = CA2C(action_size, max_action)
scores, episodes = [], []
score_avg = 0

num_episode = 1000

for e in range(num_episode):
    done = False
    score = 0
    loss_list, sigma_list = [], []
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])

    while not done:
        action = agent.get_action(state)
        next_state, reward, term, trunc, info = env.step(action)
        done = term or trunc
        next_state = np.reshape(next_state, [1, state_size])
        score += reward + 0.0025 * tf.math.cos(3 * state[0])

        loss, sigma = agent.train_model(state, action, reward, next_state, done)
        loss_list.append(loss)
        sigma_list.append(sigma)

        state = next_state

        if done:
            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print(f'episode {e} | score_avg {score_avg} | loss {np.mean(loss_list)} | sigma {np.mean(sigma_list)}')
            scores.append(score_avg)
            episodes.append(e)

            pylab.plot(episodes, scores, 'b')
            pylab.xlabel('episode')
            pylab.ylabel('score avg')
            pylab.savefig('CA2C_CartPole_graph.png')

            if score_avg >= 100:
                sys.exit()
