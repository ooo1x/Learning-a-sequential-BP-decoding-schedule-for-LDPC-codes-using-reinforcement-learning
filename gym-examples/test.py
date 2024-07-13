import gymnasium as gym
import gym_examples
import numpy as np
from agent import QLearningAgent
import time
import logging
from ray.rllib.utils.replay_buffers import ReplayBuffer
from ray.rllib.utils.replay_buffers.replay_buffer import StorageUnit
from ray.rllib.policy.sample_batch import SampleBatch
from torch.utils.tensorboard import SummaryWriter
import logging

# logging.basicConfig(filename='training_log.log',
#                     level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S')

def print_buffer(buffer):
    print("Buffer contains:")
    for idx, sample in enumerate(buffer._storage):
        print(f"Sample {idx}:")
        for key, value in sample.items():
            print(f"  {key}: {value}")

def run_episode(env, agent, buffer, batch_size = 128, render=False):
    total_reward = 0
    obs, info = env.reset()  # Unpack the tuple to get the initial state and info
    done = False
    while not done:
        action = agent.sample(obs)  # Pass only the state part of obs
        next_obs, reward, done, truncated, info = env.step(action)  # Unpack step results
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
        total_reward += reward
    return total_reward

def test_episode(env, agent):
    total_reward = 0
    obs, info = env.reset()  # Reset environment at the start of each episode
    ber_list = []
    done = False
    while not done:
        action = agent.predict(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Calculate and collect BER
        original_codeword = env.unwrapped.original_codeword
        estimate = info['estimate']
        num_errors = np.sum(np.logical_xor(original_codeword, estimate))
        ber = num_errors / len(original_codeword)
        ber_list.append(ber)

        obs = next_obs

    average_ber = np.mean(ber_list)
    return total_reward, average_ber


def main():
    writer = SummaryWriter('runs/exp')
    start_time = time.time()
    # Initialize the environment
    H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]])
    snr_db = 1.5
    env = gym.make('gym_examples/SequentialEnv-v0', H = H, snr_db = snr_db)

    # Initialize the agent
    obs_n = 2 ** H.shape[0]  # Update this based on state size calculation
    act_n = H.shape[0]  # Number of actions in environment
    agent = QLearningAgent(obs_n=obs_n, act_n=act_n, learning_rate=1e-1, gamma=0.9, e_greed=0.6)

    # Initialize replay buffer
    buffer = ReplayBuffer(capacity=10000, storage_unit=StorageUnit.TIMESTEPS)

    # Run episodes
    for episode in range(100000):
        reward = run_episode(env, agent, buffer, render=False)
        print(f"Episode {episode}: Reward: {reward}")
        writer.add_scalar("Rewards", reward, episode)

    total_duration = time.time() - start_time  # Total duration for all episodes
    print(f"Total duration: {total_duration:.1f} s")

    # Save the learned Q-table
    agent.save()

    # Define lists to hold cumulative rewards and BERs for all episodes
    total_test_rewards = []
    all_ber_list = []# List to store total rewards for each episode

    # Test
    for episode in range(20000):
        reward, average_ber = test_episode(env, agent)
        total_test_rewards.append(reward)
        all_ber_list.append(average_ber)
        #print(f"Episode {episode + 1}: Reward = {reward}, Average BER = {average_ber}")

    overall_average_ber = np.mean(all_ber_list)
    print(f"Overall Average BER {snr_db} across all episodes: {overall_average_ber}")
    env.close()
    writer.close()

if __name__ == "__main__":
    main()