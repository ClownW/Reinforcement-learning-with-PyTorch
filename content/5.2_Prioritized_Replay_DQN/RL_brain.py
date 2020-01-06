import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(1)
torch.manual_seed(1)

class SumTree(object):

	data_pointer = 0
	
	def __init__(self, capacity):
		self.capacity = capacity   # for all priority values
		self.tree = np.zeros(2 * capacity - 1)
		self.data = np.zeros(capacity, dtype=object) # for all transitions

	def add(self, p, data):
		tree_idx = self.data_pointer + self.capacity - 1
		self.data[self.data_pointer] = data # store transition in self.data
		self.update(tree_idx, p) # add p to the tree
		self.data_pointer += 1
		if self.data_pointer >= self.capacity:
			self.data_pointer = 0

	def update(self, tree_idx, p):
		change = p - self.tree[tree_idx]
		self.tree[tree_idx] = p
		while tree_idx != 0:
			tree_idx = (tree_idx - 1) // 2
			self.tree[tree_idx] += change

	def get_leaf(self, v):
		parent_idx = 0
		while True:
			cl_idx = 2 * parent_idx + 1  # left kid of the parent node
			cr_idx = cl_idx + 1
			if cl_idx >= len(self.tree):   # kid node is out of the tree, so parent is the leaf node
				leaf_idx = parent_idx
				break
			else:      # downward search, always search for a higher priority node
				if v <= self.tree[cl_idx]:
					parent_idx = cl_idx
				else:
					v -= self.tree[cl_idx]
					parent_idx = cr_idx

		data_idx = leaf_idx - self.capacity + 1
		return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

	@property
	def total_p(self):
		return self.tree[0]


class Memory(object):  # stored as (s, a, r, s_) in SumTree
	epsilon = 0.01  # small amount to avoid zero priority
	alpha = 0.6  # [0~1] convert the importance of TD error to priority
	beta = 0.4  # importance-sampling, from initial value increasing to 1
	beta_increment_per_sampling = 0.001
	abs_err_upper = 1.  # clipped abs error

	def __init__(self, capacity):
		self.tree = SumTree(capacity)

	def store(self, transition):
		max_p = np.max(self.tree.tree[-self.tree.capacity:])
		if max_p == 0:
			max_p = self.abs_err_upper
		self.tree.add(max_p, transition)  # set the max of p for new p

	def sample(self, n):
		b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
		pri_seg = self.tree.total_p / n
		self.beta = np.min([1., self.beta + self.beta_increment_per_sampling]) # max=1

		min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p   # for later calculation ISweight
		for i in range(n):
			a, b = pri_seg * i, pri_seg * (i+1)
			v = np.random.uniform(a, b)
			idx, p, data = self.tree.get_leaf(v)
			prob = p / self.tree.total_p
			ISWeights[i,0] = np.power(prob/min_prob, -self.beta)
			b_idx[i], b_memory[i,:] = idx, data
		return b_idx, b_memory, ISWeights

	def batch_update(self, tree_idx, abs_errors):
		abs_errors += self.epsilon   # convert to abs and avoid 0
		clipped_errors = np.minimum(abs_errors.data, self.abs_err_upper)
		ps = np.power(clipped_errors, self.alpha)
		for ti, p in zip(tree_idx, ps):
			self.tree.update(ti, p)


class Net(nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.el = nn.Linear(n_feature, n_hidden)
		self.q = nn.Linear(n_hidden, n_output)

	def forward(self, x):
		x = self.el(x)
		x = F.relu(x)
		x = self.q(x)
		return x


class DQNPrioritizedReplay:
	def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.005, reward_decay=0.9, e_greedy=0.9, replace_target_iter=500,
				memory_size=10000, batch_size=32, e_greedy_increment=None, output_graph=False, prioritized=True):
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

		self.prioritized = prioritized
		self.learn_step_counter = 0
		self._build_net()

		if self.prioritized:
			self.memory = Memory(capacity=memory_size)
		else:
			self.memory = np.zeros((self.memory_size, n_features*2+2))

		self.cost_his = []

	def _build_net(self):
		self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
		self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
		self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)

	def store_transition(self, s, a, r, s_):
		if self.prioritized:  # prioritized replay
			transition = np.hstack((s, [a, r], s_))
			self.memory.store(transition)  # have high priority for newly arrived transition
		else:  # random replay
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
			action = int(torch.max(actions_value, dim=1)[1])
		else:
			action = np.random.randint(0, self.n_actions)
		return action


	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.q_target.load_state_dict(self.q_eval.state_dict())
			# print("target params replaced\n")

		if self.prioritized:
			tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
			batch_memory = self.memory[sample_index, :]

		q_next, q_eval = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])), self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]))
		q_target = torch.Tensor(q_eval.data.numpy().copy())

		batch_index = np.arange(self.batch_size, dtype=np.int32)
		eval_act_index = batch_memory[:, self.n_features].astype(int)
		reward = torch.Tensor(batch_memory[:, self.n_features+1])
		q_target[batch_index, eval_act_index] = reward + self.gamma*torch.max(q_next, 1)[0]

		if self.prioritized:
			self.abs_errors = torch.sum(torch.abs(q_target-q_eval), dim=1)
			# print("ISWeights shape: ", ISWeights.shape, 'q shape: ', ((q_target-q_eval)**2), 'q: ', (q_target-q_eval))
			loss = torch.mean(torch.mean(torch.Tensor(ISWeights) * (q_target-q_eval)**2, dim=1))
			self.memory.batch_update(tree_idx, self.abs_errors)
		else:
			self.loss_func = nn.MSELoss()
			loss = self.loss_func(q_eval, q_target)

		# print("loss: ", loss, self.prioritized)
		 
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# increase epsilon
		self.cost_his.append(loss)
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1
