import gymnasium as gym
import gym_examples
import numpy as np
from agent import QLearningAgent
import time
import logging
from ray.rllib.utils.replay_buffers import ReplayBuffer
from ray.rllib.utils.replay_buffers.replay_buffer import StorageUnit
from ray.rllib.policy.sample_batch import SampleBatch
import logging

def test_episode(env, agent):
    total_reward = 0
    obs, info = env.reset()  # Reset environment at the start of each episode
    done = False
    while not done:
        action = agent.predict(obs)
        if action is None:
            obs, info = env.reset()
            agent.reset_episode()
            continue
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        obs = next_obs

    # Calculate and collect BER
    original_codeword = env.unwrapped.original_codeword
    estimate = info['estimate']
    num_errors = np.sum(np.logical_xor(original_codeword, estimate))
    print(f"num_errors: {num_errors}")
    total_bits = len(original_codeword)

    return total_reward, num_errors, total_bits


def main():
    # Initialize the environment
    H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]])
    snr_db = 0
    env = gym.make('gym_examples/SequentialEnv-v0', H = H, snr_db = snr_db)

    # Initialize the agent
    obs_n = 2 ** H.shape[0]  # Update this based on state size calculation
    act_n = H.shape[0]  # Number of actions in environment
    agent = QLearningAgent(obs_n=obs_n, act_n=act_n, learning_rate=1e-3, gamma=0.9, e_greed=0.0) #using smaller greedy
    agent.restore('q_table.npy')

    total_errors = 0
    total_bits = 0
    total_reward = 0
    counters = 0

    while total_errors < 1000:
        print(f"total_errors: {total_errors}")
        counters += 1
        reward, errors, bits = test_episode(env, agent)
        total_errors += errors
        total_bits += bits
        total_reward += reward

    average_reward = total_reward / counters
    print(f"average_reward: {average_reward}")
    ber_final = total_errors / total_bits if total_bits > 0 else 0
    print(
        f"Final BER at SNR {snr_db} dB: {ber_final}, Total bits processed: {total_bits}, Total errors: {total_errors}")

    env.close()

if __name__ == "__main__":
    main()