import gym


class Env:

	def __init__(self, envName, render_mode):
		self.envName = envName
		self.env = gym.make(envName, render_mode=render_mode)
		print("observation space: {}".format(self.env.observation_space))
		print("action space: {}".format(self.env.action_space))