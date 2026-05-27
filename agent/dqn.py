import torch
from agent.cnn import MinesweeperAgent
from agent.replaybuffer import PrioritizedReplayBuffer
from agent.preferences import AgentPreferences
from dataclasses import asdict
import numpy as np
from torch.nn import functional as F
from utils import log


class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.0001,
        epsilon=1.0,
        epsilon_min=0.1,
        gamma=0.99,
        batch_size=32,
        warmup_steps=1000,
        buffer_size=50000,
        tau=0.005,
        epsilon_decay_steps=150000,
        **kwargs
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.tau = tau
        self.H, self.W = state_dim

        self.network = MinesweeperAgent()
        self.target_network = MinesweeperAgent()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)
        self.buffer = PrioritizedReplayBuffer(state_dim, (1,), buffer_size,
                                              beta_steps=epsilon_decay_steps)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / epsilon_decay_steps

    @log
    @torch.no_grad()
    def act(self, x, training=True):
        x = self.preprocess(x)
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            a = np.random.randint(0, self.action_dim)
        else:
            x = self.get_channels(x)
            x = x.float().unsqueeze(0).to(self.device)
            q = self.network(x)
            a = self.get_action_from_response(q)
        return a

    @log
    def preprocess(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x.to(self.device)

    @log
    def learn(self):
        batch = self.buffer.sample(self.batch_size)
        s, a, r, s_prime, terminated = [x.to(self.device) for x in batch[:5]]
        weights = batch[5].to(self.device)
        leaf_indices = batch[6]

        s_channels = self.get_batch_channels(s)
        s_prime_channels = self.get_batch_channels(s_prime)

        # Double DQN: online net selects action, target net evaluates value
        with torch.no_grad():
            online_next = self.network(s_prime_channels).reshape(self.batch_size, -1)
            best_a = online_next.argmax(1, keepdim=True)
            target_next = self.target_network(s_prime_channels).reshape(self.batch_size, -1)
            next_q_max = target_next.gather(1, best_a).squeeze(1)

        td_target = r.squeeze() + (1 - terminated.squeeze()) * self.gamma * next_q_max

        current_q = self.network(s_channels).reshape(self.batch_size, -1)
        selected_q = current_q.gather(1, a.long())  # (B, 1)

        td_errors = (td_target.unsqueeze(1) - selected_q).detach().abs().reshape(-1).cpu().numpy()
        self.buffer.update_priorities(leaf_indices, td_errors)

        # IS-weighted MSE loss
        loss = (weights.unsqueeze(1) * F.mse_loss(selected_q, td_target.unsqueeze(1), reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()

        return {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }

    @log
    def process(self, transition):
        result = {}
        self.total_steps += 1

        # Convert action tuple (action_type, y, x) → flat index into Q vector
        s, a, r, s_prime, terminated = transition
        action_type, ay, ax = a
        flat_a = np.array([ay * self.W + ax if action_type == 1
                           else self.H * self.W + ay * self.W + ax])
        self.buffer.update(s, flat_a, r, s_prime, terminated)

        if self.total_steps > self.warmup_steps:
            result = self.learn()
            # Soft target network update (τ = 0.005)
            for tp, op in zip(self.target_network.parameters(), self.network.parameters()):
                tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return result

    @log
    def get_batch_channels(self, fields):
        channel0 = torch.where(fields == -3, -1.0, torch.where(fields == -2, 0.0, fields))
        channel1 = torch.where(fields == -2, 0.0, 1.0)
        channel2 = torch.where(fields == -1, 1.0, 0.0)
        return torch.stack([channel0, channel1, channel2], dim=1)

    @log
    def get_channels(self, field: torch.Tensor):
        channel0 = torch.where(
            field == -3, -1.0,
            torch.where(field == -2, 0.0, field)
        )
        channel1 = torch.where(field == -2, 0.0, 1.0)
        channel2 = torch.where(field == -1, 1.0, 0.0)
        return torch.stack([channel0, channel1, channel2], dim=0)

    @log
    def get_max_index(self, tensor: torch.Tensor):
        argmax = torch.argmax(tensor).item()
        h, w = tensor.shape
        return (argmax // w, argmax % w)

    @log
    def get_action_from_response(self, response):
        response = response[0]
        max_arg_click = self.get_max_index(response[0])
        max_arg_flag = self.get_max_index(response[1])

        use_click = torch.max(response[0]).item() >= torch.max(response[1]).item()
        max_arg = max_arg_click if use_click else max_arg_flag
        action = (int(use_click), *max_arg)
        return action


def get_agent(agent_preferences: AgentPreferences):
    return DQN(**asdict(agent_preferences))
