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
import pickle

logging.basicConfig(filename='ber_reward.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def run_episode(env, agent, buffer, batch_size, max_time_steps, render=False):
    total_reward = 0
    obs, info = env.reset()  # Unpack the tuple to get the initial state and info
    done = False
    time_step = 0
    while not done and time_step < max_time_steps:
        action = agent.sample(obs, time_step)
        if action is None:
            logging.warning(f"Action is None at timestep {time_step}")
            break
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward  # Unpack step results(just want reward)

        if info.get('msg') != 'Action already taken':
            # Create a sample transition to be stored
            sample = SampleBatch({
                "obs": [obs], "actions": [action], "rewards": [reward],
                "new_obs": [next_obs], "time_step": [time_step], SampleBatch.TERMINATEDS: [done],
                SampleBatch.TRUNCATEDS: [truncated]
            })
            buffer.add(sample)

        if len(buffer) > batch_size:
            batch = buffer.sample(batch_size)
            agent.learn_on_batch(batch)

        obs = next_obs  # Update the current state to the next one
        time_step += 1

def test_episode(env, agent):
    obs, info = env.reset()  # Reset environment at the start of each episode
    done = False
    num_errors = 0
    total_bits = 0
    time_step = 0

    agent.available_actions[:] = True

    while not done:
        action = agent.predict(obs, time_step)
        next_obs, reward, done, truncated, info = env.step(action)
        obs = next_obs

        if done:
            original_codeword = env.unwrapped.original_codeword
            estimate = info.get('estimate', [])
            num_errors = np.sum(np.logical_xor(original_codeword, estimate))
            total_bits = len(original_codeword)
            agent.available_actions[:] = True
        time_step += 1

    return reward, num_errors, total_bits

def main():
    start_time = time.time()
    # Initialize the environment
    H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]])
    snr_db = 0
    env = gym.make('gym_examples/SequentialEnv-v0', H = H, snr_db = snr_db)

    # Initialize the agent
    obs_n = 2 ** H.shape[0]  # Update this based on state size calculation
    act_n = H.shape[0]  # Number of actions in environment
    agent = QLearningAgent(obs_n=obs_n, act_n=act_n, learning_rate=1e-1, gamma=0.9, e_greed=0.6, max_time_steps=act_n)
    #agent.restore()

    # Initialize replay buffer
    buffer = ReplayBuffer(capacity=10000, storage_unit=StorageUnit.TIMESTEPS)
    batch_size = 64

    # Training phase
    for train_episodes in range(100000):
        print(f"train_episode: {train_episodes}")
        run_episode(env, agent, buffer, batch_size, act_n) #act_n is max_time step
        if (train_episodes+1) % 10000 == 0:
            # Testing phase
            agent.set_exploration(0)  # Set exploration to 0 for testing
            total_errors = 0
            total_bits = 0
            test_episodes = 0
            total_reward = 0

            while total_errors < 100:
                reward, errors, bits = test_episode(env, agent)
                test_episodes += 1
                print(f"test_episode: {test_episodes}")
                total_errors += errors
                total_bits += bits
                total_reward += reward

            ber = total_errors / total_bits if total_bits > 0 else 0
            logging.info(f"Final BER at SNR {snr_db} dB: {ber}, Total reward: {total_reward}")
            agent.set_exploration(0.6)

    agent.save()

    total_duration = time.time() - start_time
    print(f"Total duration: {total_duration:.1f} s")

    env.close()


if __name__ == "__main__":
    main()