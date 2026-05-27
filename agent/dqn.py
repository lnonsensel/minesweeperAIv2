import torch 
from agent.cnn import MinesweeperAgent
from agent.replaybuffer import ReplayBuffer
from agent.preferences import AgentPreferences
from dataclasses import asdict
import numpy as np
from torch.nn import functional as F
from utils import log
# from utils import jit_get_channels

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.00025,
        epsilon=1.0,
        epsilon_min=0.1,
        gamma=0.99,
        batch_size=4,
        warmup_steps=5000,
        buffer_size=int(1e5),
        target_update_interval=10000,
        **kwargs
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = MinesweeperAgent()
        self.target_network = MinesweeperAgent()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)
        self.buffer = ReplayBuffer(state_dim, (3, ), buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6
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
        s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
        s_channels = self.get_batch_channels(s)
        s_prime_channels = self.get_batch_channels(s_prime)

        next_q = self.target_network(s_prime_channels).detach()
        next_q_max = next_q.reshape(self.batch_size, -1).max(1)[0]

        td_target = r.squeeze() + (1 - terminated.squeeze()) * self.gamma * next_q_max

        current_q = self.network(s_channels).reshape(self.batch_size, -1)
        selected_q = current_q.gather(1, a.long())

        loss = F.mse_loss(selected_q, td_target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()
        
        result = {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }
        return result
    @log
    def process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition)

        if self.total_steps > self.warmup_steps:
            result = self.learn()
            
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.epsilon -= self.epsilon_decay
        return result
    
    @log
    def get_batch_channels(self, fields):
        channel0 = torch.where(fields == -3, -1.0, torch.where(fields == -2, 0.0, fields))
        channel1 = torch.where(fields == -2, 0.0, 1.0)
        channel2 = torch.where(fields == -1, 1.0, 0.0)
        return torch.stack([channel0, channel1, channel2], dim=1)
    @log
    def get_channels(self, field: torch.Tensor):
        '''
        Channel 0 - Поле обозрения
        Channel 1 - Клетка открыта
        Channel 2 - На клетку поставлен флаг
        '''
        channel0 = torch.where(
            field == -3, -1.0,
            torch.where(field == -2, 0.0, field)
        )
        
        channel1 = torch.where(field == -2, 0.0, 1.0)
        channel2 = torch.where(field == -1, 1.0, 0.0)
        
        channels = torch.stack([channel0, channel1, channel2], dim=0)
        return channels
    
    @log
    def get_max_index(self, tensor: torch.Tensor):
        argmax = torch.argmax(tensor).item()
        y, x = tensor.shape
        y_val = argmax // y
        x_val = argmax % x
        return (y_val, x_val)
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