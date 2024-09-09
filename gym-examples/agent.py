import numpy as np
import random
import os


class QLearningAgent(object):
    def __init__(self, obs_n, act_n, learning_rate, gamma, e_greed, max_time_steps):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.max_time_steps = max_time_steps
        self.Q_tables = {t: np.zeros((obs_n, act_n)) for t in range(max_time_steps)}
        self.available_actions = np.ones(act_n, dtype=bool)

    def set_exploration(self, e_greed):
        self.epsilon = e_greed

    def reset_episode(self):
        self.available_actions.fill(True)  # reset available actions evey episode

    def sample(self, obs, timestep):
        state = obs
        if np.all(~self.available_actions):
            self.reset_episode()
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.predict(state, timestep)  #use specific timestep qtable
        else:
            available_actions = np.where(self.available_actions)[0]
            if available_actions.size > 0:
                action = np.random.choice(available_actions)
            else:
                return None
        if action is not None:
            self.available_actions[action] = False
        return action

    def predict(self, obs, timestep):
        Q_list = self.Q_tables[timestep][obs, :] #use specific qtable
        masked_Q_list = np.where(self.available_actions, Q_list, -np.inf)
        if np.all(masked_Q_list == -np.inf):
            self.available_actions[:] = True
            masked_Q_list = np.where(self.available_actions, Q_list, -np.inf)
        maxQ = np.max(masked_Q_list)
        action_list = np.where(masked_Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        self.available_actions[action] = False
        return action

    def learn(self, obs, action, reward, next_obs, done, timestep):
        state = obs
        next_state = next_obs
        predict_Q = self.Q_tables[timestep][state, action]  #use specific qtable
        if done:
            target_Q = reward  # End of episode
        else:
            target_Q = reward + self.gamma * np.max(self.Q_tables[timestep][next_state, :])  # Bellman equation
        self.Q_tables[timestep][state, action] += self.lr * (target_Q - predict_Q)  # Update Q-table

    def learn_on_batch(self, batch):
        for i in range(len(batch["obs"])):
            obs = batch["obs"][i]
            action = batch["actions"][i]
            reward = batch["rewards"][i]
            next_obs = batch["new_obs"][i]
            done = batch["dones"][i]
            time_step = batch["time_step"][i]
            self.learn(obs, action, reward, next_obs, done, time_step)

    def save(self):
        for t, q_table in self.Q_tables.items():
            np.save(f'./q_table_step_{t}.npy', q_table)
            print(f'./q_table_step_{t}.npy saved.')

    def restore(self):
        for t in range(self.max_time_steps):
            file_path = f'./q_table_step_{t}.npy'
            if os.path.exists(file_path):
                self.Q_tables[t] = np.load(file_path)
                print(f'{file_path} loaded.')
