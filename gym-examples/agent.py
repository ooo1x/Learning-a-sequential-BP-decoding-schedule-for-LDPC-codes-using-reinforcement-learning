import numpy as np
import random


class QLearningAgent(object):
    def __init__(self,obs_n,act_n,learning_rate,gamma,e_greed):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((obs_n, act_n))

    def sample(self, obs):
        state = obs
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.predict(state)
        else:
            action = np.random.choice(self.act_n)
        return action

    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, action, reward, next_obs, done):
        state = obs
        next_state = next_obs
        predict_Q = self.Q[state, action]
        if done:
            target_Q = reward  # End of episode
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_state, :])  # Bellman equation
        self.Q[state, action] += self.lr * (target_Q - predict_Q)  # Update Q-table

    def learn_on_batch(self, batch):
        for i in range(len(batch["obs"])):
            obs = batch["obs"][i]
            action = batch["actions"][i]
            reward = batch["rewards"][i]
            next_obs = batch["new_obs"][i]
            done = batch["dones"][i]
            self.learn(obs, action, reward, next_obs, done)

    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
