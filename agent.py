import numpy as np 


class HillClimbAgent:

	def __init__(self, env):
		self.stateDim = env.env.observation_space.shape
		self.actionDim = env.env.action_space.n 
		self.buildModel()

	def buildModel(self):
		self.weights = 1e-4 * np.random.rand(*self.stateDim, self.actionDim)
		self.bestReward = -np.Inf 
		self.bestWeights = np.copy(self.weights)
		self.noise = 1e-2

	def getAction(self, state):
		probability = np.dot(np.array(state), self.weights)
		choosenAction = np.argmax(probability)
		return int(choosenAction)

	def trainModel(self, reward):
		if reward >= self.bestReward:
			self.bestReward = reward
			self.bestWeights = np.copy(self.weights)
			self.noise = max(self.noise/2, 1e-3)
		else:
			self.noise = min(self.noise*2, 2)

		self.weights = self.bestWeights + self.noise * np.random.rand(*self.stateDim, self.actionDim)