import numpy as np 


class Agent:

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
		probability = np.dot(np.array(state[0]), self.weights)
		choosenAction = np.argmax(probability)
		return choosenAction

	def trainModel(self, reward):
		if reward >= self.bestReward:
			self.bestReward = reward
			self.bestWeights = np.copy(self.weights)
			self.noise = max(self.noise/2, 1e-3)
		else:
			self.noise = min(self.noise*2, 2)

		self.weights = self.bestWeights + self.noise * np.random.rand(*self.stateDim, self.actionDim)




class QLAgent:

	def __init__(self, env, discountRate=0.97, learningRate=0.01):
		self.stateDim = env.observation_space.shape
		self.actionDim = env.action_space.n 
		self.discountRate = discountRate
		self.learningRate = learningRate
		self.eps = 1.0 
		self.buildModel()

	def buildModel(self):
		self.Q = 1e-3 * np.random.rand(*self.stateDim, self.actionDim)

	def getAction(self, state):
		Qstate = self.Q[state]
		greedyAction = np.argmax(Qstate)
		randomAction = np.random.choice(range(self.actionDim))
		return randomAction if np.random.random() < self.eps else greedyAction

	def trainModel(self, exp):
		state, action, nextState, reward, done = exp

		Qnext = self.Q[nextState] if not done else np.zeros([self.actionDim])
		Qtarget = reward + self.discountRate * np.max(Qnext)

		Qupdate = Qtarget - self.Q[state, action]
		self.Q[state, action] += self.learningRate * Qupdate

		if done:
			self.eps = self.eps * 0.99