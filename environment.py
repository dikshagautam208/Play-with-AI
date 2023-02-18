import gym


class Env:

	def __init__(self, envName):
		self.envName = envName
		self.env = gym.make(envName)
		print("observation space: {}".format(self.env.observation_space))
		print("action space: {}".format(self.env.action_space))