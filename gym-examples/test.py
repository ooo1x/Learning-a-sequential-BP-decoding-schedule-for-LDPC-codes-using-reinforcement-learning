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


def print_buffer(buffer):
    print("Buffer contains:")
    for idx, sample in enumerate(buffer._storage):
        print(f"Sample {idx}:")
        for key, value in sample.items():
            print(f"  {key}: {value}")
def run_episode(env, agent, buffer, batch_size = 1000, render=False):
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
    obs, info = env.reset()
    done = False
    ber_list = []
    step_counter = 0 
    while not done:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Calculate and collect BER
        original_codeword = env.unwrapped.original_codeword
        estimate = info['estimate']  # Get the estimate from step info
        num_errors = np.sum(original_codeword != estimate)
        ber = num_errors / len(original_codeword)
        ber_list.append(ber)

        obs = next_obs
        if done:
            print('Test reward = %.1f' % total_reward)
            print('Average BER = %f' % np.mean(ber_list))
            # Write BER results to file
            with open("ber_results.txt", "a") as file:
                 file.write(f"snr_db: {env.unwrapped.snr_db} Average BER: {np.mean(ber_list)}\n")

            break
def main():
    writer = SummaryWriter('runs/exp')
    start_time = time.time()
    # Initialize the environment
    H = np.array([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]])
    snr_db = 0.5
    env = gym.make('gym_examples/SequentialEnv-v0', H = H, snr_db = snr_db)

    # Initialize the agent
    obs_n = 2 ** H.shape[0]  # Update this based on state size calculation
    act_n = H.shape[0]  # Number of actions in environment
    agent = QLearningAgent(obs_n=obs_n, act_n=act_n, learning_rate=0.1, gamma=0.9, e_greed=0.6)

    # Initialize replay buffer
    buffer = ReplayBuffer(capacity=10000, storage_unit=StorageUnit.TIMESTEPS)

    # Run episodes
    for episode in range(100):
        reward = run_episode(env, agent, buffer, render=False)
        print(f"Episode {episode}: Reward: {reward}")
        writer.add_scalar("Rewards", reward, episode)

    total_duration = time.time() - start_time  # Total duration for all episodes
    print(f"Total duration: {total_duration:.1f} s")

    # Save the learned Q-table
    agent.save()
    test_episode(env, agent)
    env.close()
    writer.close()

if __name__ == "__main__":
    main()
    