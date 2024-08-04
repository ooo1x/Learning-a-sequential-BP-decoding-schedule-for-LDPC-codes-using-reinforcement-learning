import numpy as np
import random


class QLearningAgent(object):
    def __init__(self,obs_n,act_n,learning_rate,gamma,e_greed):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((obs_n, act_n))
        self.available_actions = np.ones(act_n, dtype=bool)

    def set_exploration(self, e_greed):
        self.epsilon = e_greed

    def reset_episode(self):
        self.available_actions.fill(True)

    def sample(self, obs):
        state = obs
        if np.all(~self.available_actions):
            self.reset_episode()
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.predict(state)
        else:
            available_actions = np.where(self.available_actions)[0]
            if available_actions.size > 0:
                action = np.random.choice(available_actions)
            else:
                return None
        if action is not None:
            self.available_actions[action] = False
        return action


    def predict(self, obs):
        Q_list = self.Q[obs, :]
        masked_Q_list = np.where(self.available_actions, Q_list, -np.inf)
        if np.all(masked_Q_list == -np.inf):
            self.available_actions[:] = True
            masked_Q_list = np.where(self.available_actions, Q_list, -np.inf)
        maxQ = np.max(masked_Q_list)
        action_list = np.where(masked_Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        self.available_actions[action] = False
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
