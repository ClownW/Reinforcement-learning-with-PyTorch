from maze_env import Maze
from RL_brain import QLearningTable

def update():
	for episode in range(150):
		observation = env.reset()
		print(episode)
		while True:
			env.render()
			action = RL.choose_action(str(observation))
			# print("observation: {}".format(observation))
			observation_, reward, done = env.step(action)
			RL.learn(str(observation), action, reward, str(observation_))
			# print(RL.q_table)
			observation = observation_
			if done:
				break
	print('game over')
	env.destroy()

if __name__ == '__main__':
	env = Maze()
	# print("env.n_actions: {}".format(env.n_actions))
	RL = QLearningTable(actions=list(range(env.n_actions)))

	env.after(100, update)
	env.mainloop()