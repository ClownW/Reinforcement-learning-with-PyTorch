import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(1)
torch.manual_seed(1)


class Net(nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.layer = nn.Linear(n_feature, n_hidden)
		self.all_act = nn.Linear(n_hidden, n_output)

	def forward(self, x):
		x = self.layer(x)
		x = torch.tanh(x)
		x = self.all_act(x)
		return x




class PolicyGradient:
	def __init__(self, n_actions, n_features, n_hidden=10, learning_rate=0.01, reward_decay=0.95):
		self.n_actions = n_actions
		self.n_features = n_features
		self.n_hidden = n_hidden
		self.lr = learning_rate
		self.gamma = reward_decay

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []

		self._build_net()

	def _build_net(self):
		self.net = Net(self.n_features, self.n_hidden, self.n_actions)
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

	def choose_action(self, observation):
		observation = torch.Tensor(observation[np.newaxis, :])
		prob_weights = self.net(observation)
		prob = F.softmax(prob_weights)
		action = np.random.choice(range(prob_weights.shape[1]), p=prob.data.numpy().ravel())
		return action

	def store_transition(self, s, a, r):
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)

	def learn(self):
		# discount and normalize episode reward
		discounted_ep_rs_norm = self._discount_and_norm_rewards()
		obs = torch.Tensor(np.vstack(self.ep_obs))
		acts = torch.Tensor(np.array(self.ep_as))
		vt = torch.Tensor(discounted_ep_rs_norm)

		all_act = self.net(obs)

		# cross_entropy combines nn.LogSoftmax() and nn.NLLLoss() in one single class
		neg_log_prob = F.cross_entropy(all_act, acts.long(), reduce=False)
		loss = torch.mean(neg_log_prob * vt)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []
		return discounted_ep_rs_norm

	def _discount_and_norm_rewards(self):
		discounted_ep_rs = np.zeros_like(self.ep_rs)
		running_add = 0
		for t in reversed(range(len(self.ep_rs))):
			running_add = running_add*self.gamma + self.ep_rs[t]
			discounted_ep_rs[t] = running_add

		discounted_ep_rs -= np.mean(discounted_ep_rs)
		discounted_ep_rs /= np.std(discounted_ep_rs)
		return discounted_ep_rs
