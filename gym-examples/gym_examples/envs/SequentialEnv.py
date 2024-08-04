import gymnasium as gym
from gymnasium import spaces
import numpy as np
from algorithm import BeliefPropagation
from datetime import datetime
from codeword_generator import generate_random_codewords, h2g
import random
from scipy.sparse import coo_matrix, csr_matrix


class SequentialEnv(gym.Env):
    def __init__(self, H, snr_db, max_iter=1, sequence=None):
        print("SequentialEnv init...")
        super(SequentialEnv, self).__init__()
        if isinstance(H, np.ndarray):
            self.H = csr_matrix(H)
        elif isinstance(H, coo_matrix):
            self.H = csr_matrix(H)
        else:
            self.H = H
        self.G = h2g(H)
        self.snr_db = snr_db
        self.max_iter = max_iter
        self.sequence = sequence if sequence is not None else list(range(H.shape[0]))
        self.bp_decoder = BeliefPropagation(self.H, self.max_iter, self.sequence)
        self.channel_llr = np.zeros(self.H.shape[1])
        self.current_llr = np.zeros(self.H.shape[1])
        self.original_codeword = np.zeros(self.H.shape[1])
        self.observation_space = spaces.Discrete(2 ** H.shape[0]) #state 000,001,010,100,101,110,111
        self.action_space = spaces.Discrete(H.shape[0])#action 0,1,2
        self.cn_updated = np.zeros(H.shape[0], dtype=bool) # Tracks whether each CN has been updated
        self.step_counter = 0 #use it to control channel_llr
        self.messages = np.zeros((self.H.shape[0], self.H.shape[1]))

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._generate_llr()
        self.generate_initial_codeword()
        self.state = self._get_state(self.channel_llr)
        self.cn_updated.fill(False)  # Reset the update status for each check node
        return self.state, {}

    def generate_initial_codeword(self):
        original_codewords = generate_random_codewords(self.G, 1)
        self.original_codeword = random.choice(original_codewords)

    def _generate_llr(self):
        transmitted_codeword = 1 - 2 * self.original_codeword
        snr_linear = 10 ** (self.snr_db / 10)
        Eb = 1
        N0 = Eb / snr_linear
        sigma = np.sqrt(N0 / 2)
        received_codeword = transmitted_codeword + sigma * np.random.randn(len(self.original_codeword))
        self.channel_llr = 2 * received_codeword / (sigma ** 2)#awgn channel
        # print("generated llr", self.channel_llr)
        self.current_llr = np.copy(self.channel_llr)

    def _get_state(self, updated_llr):
        check_node = self.H.dot(updated_llr)
        binary_string = ''.join('0' if x >= 0 else '1' for x in check_node) #get check node(if <0 then1; else 0)
        state = int(binary_string, 2)
        return state

    def step(self, action):
        if self.cn_updated[action]:
            return self.state, 0, False, False, {'msg': 'Action already taken'}
        if self.step_counter % 4 == 0:
            self._generate_llr()
        self.current_llr, residuals = self.bp_decoder.decode(self.current_llr, self.sequence[action])
        reward = self._compute_reward(residuals)
        print("REWARD", reward)
        self.cn_updated[action] = True
        done = np.all(self.cn_updated)
        info = {}
        if done:
            # If all check nodes have been updated, reset for next round
            estimate = np.array([1 if x < 0 else 0 for x in self.current_llr])
            info = {'estimate': estimate}
            self.cn_updated.fill(False)  # Reset for the next set of updates

        self.state = self._get_state(self.current_llr)
        truncated = False

        self.current_step += 1
        self.step_counter += 1
        return self.state, reward, done, truncated, info

    def _compute_reward(self, residuals):
        max_residual = np.max(residuals)
        return max_residual
#different reward function

    def render(self):
        pass

    def close(self):
        pass

