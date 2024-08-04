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

logging.basicConfig(filename='rewards_average.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def run_episode(env, agent, buffer, batch_size, render=False):
    total_reward = 0
    obs, info = env.reset()  # Unpack the tuple to get the initial state and info
    done = False
    while not done:
        action = agent.sample(obs)
        print("action", action)
        if action is None:
            obs, info = env.reset()
            agent.reset_episode()
            continue
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward# Unpack step results
        sample = SampleBatch({
            "obs": [obs], "actions": [action], "rewards": [reward],
            "new_obs": [next_obs], SampleBatch.TERMINATEDS: [done],
            SampleBatch.TRUNCATEDS: [truncated]
        })
        buffer.add(sample)
        #print_buffer(buffer)
        if len(buffer) > batch_size:
            batch = buffer.sample(batch_size)
            agent.learn_on_batch(batch)

        obs = next_obs  # Update the current state to the next one
    return total_reward

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
    total_bits = len(original_codeword)

    return total_reward, num_errors, total_bits

def main():
    start_time = time.time()
    # Initialize the environment
    H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]])
    snr_db = 0
    env = gym.make('gym_examples/SequentialEnv-v0', H = H, snr_db = snr_db)

    # Initialize the agent
    obs_n = 2 ** H.shape[0]  # Update this based on state size calculation
    act_n = H.shape[0]  # Number of actions in environment
    agent = QLearningAgent(obs_n=obs_n, act_n=act_n, learning_rate=1e-1, gamma=0.9, e_greed=0.6)
    agent.restore('q_table.npy')

    # Initialize replay buffer
    buffer = ReplayBuffer(capacity=10000, storage_unit=StorageUnit.TIMESTEPS)
    batch_size = 64

    # Training phase
    for episode in range(5):
        run_episode(env, agent, buffer, batch_size)

    agent.save()

    # # Testing phase
    # agent.set_exploration(0.0)  # Set exploration to 0 for testing
    # total_errors = 0
    # total_bits = 0
    # while total_errors < 1000:
    #     reward, errors, bits = test_episode(env, agent)
    #     total_errors += errors
    #     total_bits += bits
    #     if total_errors >= 1000:
    #         break  # Exit the loop once 1000 errors are accumulated
    #
    # ber = total_errors / total_bits if total_bits > 0 else 0
    # print(f"Final BER at SNR {snr_db} dB: {ber}, Total bits processed: {total_bits}, Total errors: {total_errors}")
    # total_duration = time.time() - start_time
    # print(f"Total duration: {total_duration:.1f} s")

    env.close()


if __name__ == "__main__":
    main()