import gym
import pygame
import tqdm

env = gym.make("CartPole-v1", render_mode="human")

print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

env.reset()

done = False
total_reward = 0
while not done:
   env.render()
   obs, rew, d1, d2, info = env.step(env.action_space.sample())
   done = d1 or d2
   total_reward += rew
   print(f"{obs} -> {rew}")
print(f"Total reward: {total_reward}")

env.close()


env = gym.make("CartPole-v1", render_mode="human")
env.reset()

done = False
total_reward = 0
while not done:
   env.render()
   obs, rew, d1, d2, info = env.step(env.action_space.sample())
   done = d1 or d2
   total_reward += rew
   print(f"{obs} -> {rew}")
print(f"Total reward: {total_reward}")

env.close()