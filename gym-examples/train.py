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


def main():
    start_time = time.time()
    # Initialize the environment
    H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]])
    snr_db = 0
    env = gym.make('gym_examples/SequentialEnv-v0', H = H, snr_db = snr_db)

    # Initialize the agent
    obs_n = 2 ** H.shape[0]  # Update this based on state size calculation
    act_n = H.shape[0]  # Number of actions in environment
    agent = QLearningAgent(obs_n=obs_n, act_n=act_n, learning_rate=1e-3, gamma=0.9, e_greed=0.6)
    agent.restore('q_table.npy')

    # Initialize replay buffer
    buffer = ReplayBuffer(capacity=10000, storage_unit=StorageUnit.TIMESTEPS)
    batch_size = 64
    total_rewards = []

    # Run episodes
    for episode in range(1000):
        print('Episode:', episode)
        reward = run_episode(env, agent, buffer, batch_size, render=False)
        total_rewards.append(reward)

        # Log average reward every 1000 episodes
        if (episode + 1) % 1000 == 0:
            average_reward = sum(total_rewards) / len(total_rewards)
            logging.info(f"Episode {episode + 1}: Average Reward: {average_reward}")
            print(f"Episode {episode + 1}: Average Reward: {average_reward}")
            total_rewards = []

    total_duration = time.time() - start_time  # Total duration for all episodes
    print(f"Total duration: {total_duration:.1f} s")

    # Save the learned Q-table
    agent.save()

    env.close()

if __name__ == "__main__":
    main()