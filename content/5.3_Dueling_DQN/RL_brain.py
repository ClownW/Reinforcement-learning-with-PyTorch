import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(1)
torch.manual_seed(1)

class Net(nn.Module):
	def __init__(self, n_feature, n_hidden, n_output, dueling=False):
		super(Net, self).__init__()
		self.dueling = dueling
		self.l1 = nn.Linear(n_feature, n_hidden)
		if self.dueling:
			self.values = nn.Linear(n_hidden, 1)
			self.advantages = nn.Linear(n_hidden, n_output)
		else:
			self.q = nn.Linear(n_hidden, n_output)

	def forward(self, x):
		x = self.l1(x)
		x = F.relu(x)
		if self.dueling:
			value = self.values(x)
			advantages = self.advantages(x)
			out = value + (advantages-torch.mean(advantages, dim=1, keepdim=True))
		else:
			out = self.q(x)
		return out


class DuelingDQN:
	def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.001, reward_decay=0.9, e_greedy=0.9, 
				replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=None, dueling=True):
		self.n_actions = n_actions
		self.n_features = n_features 
		self.n_hidden = n_hidden
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
		self.dueling = dueling

		self.learn_step_counter = 0
		self.memory = np.zeros((self.memory_size, n_features*2+2))
		self._build_net()
		self.cost_his = []

	def _build_net(self):
		self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions, self.dueling)
		self.q_target = Net(self.n_features, self.n_hidden, self.n_actions, self.dueling)

		self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)
		self.loss_func = nn.MSELoss()

	def store_transition(self, s, a, r, s_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0
		transition = np.hstack((s, [a, r], s_))
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition
		self.memory_counter += 1

	def choose_action(self, observation):
		observation = torch.Tensor(observation[np.newaxis, :])
		if np.random.uniform() < self.epsilon:
			actions_value = self.q_eval(observation)
			action = torch.max(actions_value, dim=1)[1]
		else:
			action = np.random.randint(0, self.n_actions)
		return action

	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.q_target.load_state_dict(self.q_eval.state_dict())

		sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]

		q_next, q_eval = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])), self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]))
		q_target = torch.Tensor(q_eval.data.numpy().copy())

		batch_index = np.arange(self.batch_size, dtype=np.int32)
		eval_act_index = batch_memory[:, self.n_features].astype(int)
		reward = torch.Tensor(batch_memory[:, self.n_features+1])

		q_target[batch_index, eval_act_index] = reward + self.gamma*torch.max(q_next, dim=1)[0]

		loss = self.loss_func(q_eval, q_target)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.cost_his.append(loss)

		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1