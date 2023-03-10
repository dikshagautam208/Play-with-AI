import sys
from environment import *
from agent import *
import matplotlib.pyplot as plt 
from matplotlib import style
style.use("ggplot")


numEpisodes = 200


def drawPlots(rewards):
	plt.figure("Reward Trend")
	plt.plot(rewards, color = 'b')
	plt.xlim(-5, len(rewards) + 10)
	plt.ylim(-5, max(rewards) + 10)
	plt.show()


def main(env, agent):
	rewardTrend = []

	for ep in range(numEpisodes):
		state, _ = env.env.reset()
		epReward = 0
		done = False
		while not done:
			action = agent.getAction(state)
			state, reward, terminated, truncated, info = env.env.step(action)
			done = terminated or truncated
			env.env.render()
			epReward += reward

		agent.trainModel(epReward)
		rewardTrend.append(epReward)
		print("episode: {}, reward: {:.2f}".format(ep, epReward))

	env.env.close()
	drawPlots(rewardTrend)


if __name__ == "__main__":
	if len(sys.argv) > 2:
		env = Env("CartPole-v1", "human")
		numEpisodes = int(sys.argv[1])
	elif len(sys.argv) > 1:
		env = Env("CartPole-v1", "rgb_array")
		numEpisodes = int(sys.argv[1])
	else:
		env = Env("CartPole-v1", "rgb_array")
	agent = HillClimbAgent(env)
	main(env, agent)