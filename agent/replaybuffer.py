import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a, r, s_prime, terminated):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.s_prime[ind]),
            torch.FloatTensor(self.terminated[ind]),
            torch.ones(batch_size),   # uniform IS weights
            ind.tolist(),             # leaf indices (data indices for uniform buffer)
        )

    def update_priorities(self, indices, td_errors):
        pass  # no-op for uniform buffer


class SumTree:
    """Binary sum tree for O(log n) priority sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def update(self, leaf_idx, priority):
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        idx = leaf_idx
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        return idx

    @property
    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """Replay buffer that samples transitions proportional to their TD error.

    alpha controls how much prioritization is used (0 = uniform).
    beta is the IS weight exponent, annealed from beta_start to 1.0 over beta_steps.
    """

    def __init__(self, state_dim, action_dim, max_size=int(1e5),
                 alpha=0.6, beta_start=0.4, beta_steps=150000):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_steps
        self.eps = 1e-6
        self.max_priority = 1.0  # max of priority^alpha seen so far

        self.tree = SumTree(max_size)
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def update(self, s, a, r, s_prime, terminated):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated
        leaf_idx = self.ptr + self.tree.capacity - 1
        self.tree.update(leaf_idx, self.max_priority)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        leaf_indices, data_indices, priorities = [], [], []
        segment = self.tree.total / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            leaf_idx = self.tree.get(s)
            data_idx = int(np.clip(leaf_idx - (self.tree.capacity - 1), 0, self.size - 1))
            leaf_indices.append(leaf_idx)
            data_indices.append(data_idx)
            priorities.append(self.tree.tree[leaf_idx])

        probs = np.array(priorities) / (self.tree.total + self.eps)
        weights = (self.size * probs + self.eps) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)

        return (
            torch.FloatTensor(self.s[data_indices]),
            torch.FloatTensor(self.a[data_indices]),
            torch.FloatTensor(self.r[data_indices]),
            torch.FloatTensor(self.s_prime[data_indices]),
            torch.FloatTensor(self.terminated[data_indices]),
            torch.FloatTensor(weights),
            leaf_indices,
        )

    def update_priorities(self, leaf_indices, td_errors):
        for leaf_idx, td_err in zip(leaf_indices, td_errors):
            priority = (float(abs(td_err)) + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(leaf_idx, priority)
