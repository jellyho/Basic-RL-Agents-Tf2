import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import pylab
import gym

class ContinuousA2C:
    def __init__(self, state_size, action_size, max_action):
        self.action_size = action_size
        self.state_size = state_size
        self.max_action = max_action

        self.std_bound = [1e-2, 1.0]
        self.discount_factor = 0.95
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.001

        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_learning_rate)

    def build_critic(self):
        critic_input = tf.keras.Input((self.state_size,))
        critic = tf.keras.layers.Dense(64, activation='relu')(critic_input)
        critic = tf.keras.layers.Dense(32, activation='relu')(critic)
        critic = tf.keras.layers.Dense(16, activation='relu')(critic)
        critic_out = tf.keras.layers.Dense(1, activation='linear')(critic)

        return tf.keras.Model(critic_input, critic_out)

    def build_actor(self):
        actor_input = tf.keras.Input((self.state_size,))
        actor = tf.keras.layers.Dense(64, activation='relu')(actor_input)
        actor = tf.keras.layers.Dense(32, activation='relu')(actor)
        actor = tf.keras.layers.Dense(16, activation='relu')(actor)

        mu_out = tf.keras.layers.Dense(self.action_size, activation='tanh')(actor)
        mu_out = tf.keras.layers.Lambda(lambda x: x * self.max_action)(mu_out)

        sigma_out = tf.keras.layers.Dense(self.action_size, activation='softplus')(actor)

        return tf.keras.Model(actor_input, [mu_out, sigma_out])

    def get_action(self, state):
        mu, sigma = self.actor(state)
        sigma = tf.clip_by_value(sigma, self.std_bound[0], self.std_bound[1])

        dist = tfd.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0]
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    def train_model(self, state, action, reward, next_state, done):
        actor_params = self.actor.trainable_variables
        critic_params = self.critic.trainable_variables

        next_value = self.critic(next_state)
        target = reward + (1 - done) * self.discount_factor * next_value[0]

        with tf.GradientTape() as tape1:
            mu, sigma = self.actor(state, training=True)
            adv = tf.stop_gradient(target - self.critic(state, training=True))
            dist = tfd.Normal(loc=mu, scale=sigma)
            action_prob = dist.prob([action])[0]
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            actor_loss = tf.reduce_mean(cross_entropy * adv)

        actor_grads = tape1.gradient(actor_loss, actor_params)
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_params))

        with tf.GradientTape() as tape2:
            value = self.critic(state, training=True)
            critic_loss = 0.5 * tf.square(target - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

        critic_grads = tape2.gradient(critic_loss, critic_params)
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_params))

        return actor_loss, critic_loss, sigma

    def load_weights(self, path):
        self.actor.load_weights(path + 'pendulum_actor.h5')
        self.critic.load_weights(path + 'pendulum_critic.h5')

    def save_weights(self, path):
        self.actor.save_weights(path + 'pendulum_actor.h5')
        self.critic.save_weights(path + 'pendulum_critic.h5')

    def train(self, env, num_episode=1000):
        scores, episodes = [], []
        score_avg = 0

        for e in range(num_episode):
            done = False
            score = 0
            actor_loss_list, critic_loss_list, sigma_list = [], [], []

            state, info = env.reset()
            state = np.reshape(state, [1, self.state_size])

            while not done:
                action = self.get_action(state)
                next_state, reward, term, trunc, info = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                reward = (reward + 8) / 8

                done = term or trunc

                score += reward

                actor_loss, critic_loss, sigma = self.train_model(state, action, reward, next_state, done)
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
                sigma_list.append(sigma)

                state = next_state

                if done:
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    print("episode: {:3d} | score avg: {:3.2f} | actor_loss: {:.3f} | critic_loss: {:.3f} | sigma: {:.3f}".format(
                        e, score_avg, np.mean(actor_loss_list), np.mean(critic_loss_list), np.mean(sigma_list)))

                    scores.append(score_avg)
                    episodes.append(e)
                    pylab.plot(episodes, scores, 'b')
                    pylab.xlabel("episode")
                    pylab.ylabel("average score")
                    pylab.savefig("graph.png")

                    if e % 100 == 0:
                        self.save_weights('')


env = gym.make('BipedalWalker-v3', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
max_action = env.action_space.high[0]
agent = ContinuousA2C(state_size, action_size, max_action)
#agent.load_weights('')
agent.train(env)
