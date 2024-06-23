import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Matrix_version.Matrix_layered.algorithm import BeliefPropagation

class SequentialEnv(gym.Env):
    def __init__(self, llr_file, H, max_iter=10,sequence=None):
        super(SequentialEnv, self).__init__()
        self.llr = np.loadtxt(llr_file)
        self.H = H
        self.current_step = 0
        self.max_iter = max_iter
        self.sequence = sequence if sequence is not None else list(range(H.shape[1]))
        self.bp_decoder = BeliefPropagation(self.H, self.max_iter, self.sequence)

        self.observation_space = spaces.Discrete(2 ** H.shape[1]) #state 000,001,010,100,101,110,111
        self.action_space = spaces.Discrete(3)#action 0,1,2
        self.state = self._get_state()

    def reset(self):
        self.current_step = 0
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        llr_value = self.llr[self.current_step]
        check_node = np.dot(self.H, llr_value) % 2
        state = int(''.join(map(str, check_node)), 2)
        return state

    def step(self, action):
        self._update_llr(action)
        self.current_step += 1
        done = self.current_step >= len(self.llr)
        reward = self._compute_reward()
        self.state = self._get_state()
        return self.state, reward, done, {}

    def _update_llr(self, action):
        llr_value = self.llr[self.current_step]
        selected_cn_indices = [self.sequence[action]]
        _, updated_llr, _ = self.bp_decoder.decode(llr_value, selected_cn_indices)
        self.llr[self.current_step] = updated_llr

    def _compute_reward(self):
        if self.current_step == 0:
            return 0

        transmitted_bits = np.array([1 if llr < 0 else 0 for llr in self.llr[self.current_step]])
        reconstructed_bits = np.array([1 if llr < 0 else 0 for llr in self.llr[self.current_step - 1]])
        la = len(transmitted_bits)
        correct_bits = np.sum(transmitted_bits == reconstructed_bits)
        reward = correct_bits / la

        return reward

    def render(self, mode='human'):
        pass

    def close(self):
        pass