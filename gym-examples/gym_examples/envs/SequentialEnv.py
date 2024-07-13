import gymnasium as gym
from gymnasium import spaces
import numpy as np
from algorithm import BeliefPropagation
import logging
from datetime import datetime
from codeword_generator import generate_random_codewords, h2g, row_rank

class SequentialEnv(gym.Env):
    def __init__(self, H, snr_db, max_iter=10, sequence=None):
        print("SequentialEnv init...")
        super(SequentialEnv, self).__init__()
        self.H = H
        self.G = h2g(H)
        self.snr_db = snr_db
        self.max_iter = max_iter
        self.sequence = sequence if sequence is not None else list(range(H.shape[0]))
        self.bp_decoder = BeliefPropagation(self.H, self.max_iter, self.sequence)
        self.llr = None
        self.original_codeword = None
        self.observation_space = spaces.Discrete(2 ** H.shape[0]) #state 000,001,010,100,101,110,111
        self.action_space = spaces.Discrete(H.shape[0])#action 0,1,2

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"log_{current_time}.log"
        logging.basicConfig(filename=log_filename,
                            filemode='w',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._generate_llr() 
        self.state = self._get_state(self.llr)
        return self.state, {}

    def _generate_llr(self):
        self.original_codeword = generate_random_codewords(self.G, 1)[0]
        transmitted_codeword = 1 - 2 * self.original_codeword
        snr_linear = 10 ** (self.snr_db / 10)
        sigma = np.sqrt(1 / (2 * snr_linear))
        received_codeword = transmitted_codeword + sigma * np.random.randn(len(self.original_codeword))
        self.llr = 2 * received_codeword / (sigma ** 2)#awgn channel
        # print("generated llr", self.llr)

    def _get_state(self,updated_llr):
        check_node = np.dot(self.H, updated_llr)
        binary_string = ''.join('0' if x >= 0 else '1' for x in check_node) #get check node(if <0 then1; else 0)
        state = int(binary_string, 2)
        return state

    def step(self, action):
        selected_cn_indices = [self.sequence[action]]
        updated_llr, residuals, estimate = self.bp_decoder.decode(self.llr, selected_cn_indices)
        reward = self._compute_reward(residuals)
        self.state = self._get_state(updated_llr)
        info = {'estimate': estimate}
        done = False  
        truncated = False  
        if self.current_step >= 25 -1:  # max_step in one episode
            done = True
            truncated = False
        self.current_step += 1
        logging.info(f"Step: {self.current_step}, Action: {action}, Reward: {reward}, State: {self.state}")
        return self.state, reward, done, truncated, info

    def _compute_reward(self, residuals):
        max_residual = np.max(residuals)
        return  max_residual
#different reward function

    def render(self):
        pass

    def close(self):
        pass