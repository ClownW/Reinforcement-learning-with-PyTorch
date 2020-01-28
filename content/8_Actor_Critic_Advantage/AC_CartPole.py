import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


np.random.seed(1)
torch.manual_seed(1)


MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200    # renders environment if total episode reward is greater than this threshold
MAX_EP_STEPS = 1000   # maximum time steps in one episode
RENDER = False   # rendering wastes time
GAMMA = 0.9   # reward discount in TD error
LR_A = 0.001   # learning rate for actor
LR_C = 0.01   # learning rete for critic


env = gym.make('CartPole-v0')
env.seed(1)   # reproducible
env = env.unwrapped


N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class Net(nn.Module):
	def __init__(self, n_feature, n_hidden, n_output, activate=False):
		super(Net, self).__init__()
		self.l1 = nn.Linear(n_feature, n_hidden)
		self.acts_prob = nn.Linear(n_hidden, n_output)
		self.activate=activate


	def forward(self, x):
		x = self.l1(x)
		x = F.relu(x)
		x = self.acts_prob(x)
		if self.activate:
			x = F.softmax(x)
		return x


class Actor(object):
	def __init__(self, n_features, n_actions, n_hidden=20, lr=0.001):
		self.n_features = n_features
		self.n_actions = n_actions
		self.n_hidden = n_hidden
		self.lr = lr

		self._build_net()


	def _build_net(self):
		self.actor_net = Net(self.n_features, self.n_hidden, self.n_actions, activate=True)
		self.optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.lr)


	def choose_action(self, s):
		s = torch.Tensor(s[np.newaxis, :])
		probs = self.actor_net(s)
		return np.random.choice(np.arange(probs.shape[1]), p=probs.data.numpy().ravel())


	def learn(self, s, a, td):
		s = torch.Tensor(s[np.newaxis, :])
		acts_prob = self.actor_net(s)
		log_prob = torch.log(acts_prob[0, a])
		exp_v = torch.mean(log_prob * td)

		loss = -exp_v
		self.optimizer.zero_grad()
		loss.backward(retain_graph=True)
		self.optimizer.step()

		return exp_v


class Critic(object):
	def __init__(self, n_features, lr=0.01):
		self.n_features = n_features
		self.lr = lr

		self._build_net()


	def _build_net(self):
		self.critic_net = Net(self.n_features, 20, 1)
		self.optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr)


	def learn(self, s, r, s_):
		s, s_ = torch.Tensor(s[np.newaxis, :]), torch.Tensor(s_[np.newaxis, :])
		v, v_ = self.critic_net(s), self.critic_net(s_)
		td_error = r + GAMMA * v_ - v
		loss = td_error ** 2

		self.optimizer.zero_grad()
		loss.backward(retain_graph=True)
		self.optimizer.step()

		return td_error



actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(n_features=N_F, lr=LR_C)   # we need a good teacher, so the teacher should learn faster than the actor

for i_episode in range(MAX_EPISODE):
	s = env.reset()
	t = 0
	track_r = []

	while True:
		if RENDER: env.render()

		a = actor.choose_action(s)

		s_, r, done, info = env.step(a)

		if done: r = -20

		track_r.append(r)

		td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
		actor.learn(s, a, td_error)   # true_gradient = grad[logPi(s, a) * td_error]

		s = s_
		t += 1

		if done or t>=MAX_EP_STEPS:
			ep_rs_sum = sum(track_r)

			if 'running_reward' not in globals():
				running_reward = ep_rs_sum
			else:
				running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

			# if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True
			print("episode: ", i_episode, "  reward:", int(running_reward))
			break