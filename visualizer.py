from environment import *
from agent import *

numEpisodes = 200


def main(env, agent):
	rewardTrend = []

	for ep in range(numEpisodes):
		state = env.env.reset()
		epReward = 0
		done = False
		while not done:
			action = agent.getAction(state)
			state, reward, done, info = env.env.step(action)[0]
			env.env.render()
			epReward += reward

		agent.trainModel(epReward)
		print("episode: {}, reward: {:.2f}".format(ep, epReward))

		env.env.close()


if __name__ == "__main__":
	env = Env("CartPole-v0")
	agent = Agent(env)
	main(env, agent)