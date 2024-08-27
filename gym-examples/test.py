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
    obs, info = env.reset()  # Reset environment at the start of each episode
    done = False
    num_errors = 0
    total_bits = 0

    agent.available_actions[:] = True

    while not done:
        action = agent.predict(obs)
        print(f"action: {action}, obs:{obs}")
        next_obs, reward, done, truncated, info = env.step(action)
        obs = next_obs
        print(f"next_obs: {obs}")

        if done:
            original_codeword = env.unwrapped.original_codeword
            estimate = info.get('estimate', [])
            num_errors = np.sum(np.logical_xor(original_codeword, estimate))
            total_bits = len(original_codeword)
            agent.available_actions[:] = True

    return reward, num_errors, total_bits


def main():
    # Initialize the environment
    H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]])
    snr_db = 0
    env = gym.make('gym_examples/SequentialEnv-v0', H = H, snr_db = snr_db)

    # Initialize the agent
    obs_n = 2 ** H.shape[0]  # Update this based on state size calculation
    act_n = H.shape[0]  # Number of actions in environment
    agent = QLearningAgent(obs_n=obs_n, act_n=act_n, learning_rate=1e-1, gamma=0.9, e_greed=0.0) #using smaller greedy
    agent.restore('q_table.npy')

    total_errors = 0
    total_bits = 0
    total_reward = 0
    test_episodes = 0

    while total_errors < 1:
        reward, errors, bits = test_episode(env, agent)
        test_episodes += 1
        print(f"test_episode: {test_episodes}")
        total_errors += errors
        total_bits += bits
        total_reward += reward

    ber = total_errors / total_bits if total_bits > 0 else 0
    logging.info(f"Final BER at SNR {snr_db} dB: {ber}, Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()

