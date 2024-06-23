import gymnasium as gym
import gym_examples
import numpy as np

llr_file = 'llrs_snr_2.5.txt'
H = np.array([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]])
env = gym.make('gym_examples/SequentialEnv-v0', H = H, llr_file = llr_file)

observation = env.reset()
action = env.action_space.sample()
observation, reward, done, info = env.step(action)

print("Observation:", observation)
print("Reward:", reward)
print("Done:", done)
print("Info:", info)

env.close()
