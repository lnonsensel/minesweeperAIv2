import csv
from agent.dqn import DQN
from minesweeper_env.minenv import get_minenv
from minesweeper_env.preferences import MinesweeperEnvPreferences
from agent.preferences import AgentPreferences
from agent.dqn import get_agent
import numpy as np
import os
import time
import datetime
import torch
import matplotlib.pyplot as plt
from teacher.preferences import TeacherPreferences
from teacher.config import MODELS_CHECKPOINTS_PATH
from dataclasses import asdict
import typing as tp
from utils import log

class Teacher:
    def __init__(self,
                 agent_preferences: AgentPreferences,
                 env_preferences: MinesweeperEnvPreferences,
                 eval_interval: int,
                 learning_max_steps: int,
                 model_filename: str,
                 resume_from: str | None = None,
                 use_tensorboard: bool = False) -> None:
        self.env_preferences = env_preferences
        self.agent_preferences = agent_preferences
        self.env = get_minenv(env_preferences)
        self.agent_preferences.action_dim = self.env.action_space.n
        field_size = self.env_preferences.game_preferences.field_size
        self.agent_preferences.state_dim = field_size
        self.agent = get_agent(agent_preferences)
        self.agent.action_dim = 2 * field_size[0] * field_size[1]
        self.top_eval_score = -float('inf')
        self.last_eval_scores = None
        self.last_eval_rewards = None
        self.last_win_rate = 0.0
        self.last_reward = None
        self.last_saved_model_name = None
        self.is_warmup = True
        self.non_warmup_start_time = None
        self.eval_interval = eval_interval
        self.learning_max_steps = learning_max_steps
        self.model_filename = model_filename
        self.last_action = None
        self.evals_counter = 0
        self.use_tensorboard = use_tensorboard
        self.last_avg_return = 0.0
        self.last_loss = 0.0
        self.recent_eval_returns: list[float] = []
        self._last_print_time: float = 0.0

        if resume_from:
            checkpoint = torch.load(f'{MODELS_CHECKPOINTS_PATH}/{resume_from}', weights_only=False)
            self.agent.network.load_state_dict(checkpoint['network'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
            self.agent.total_steps = checkpoint['total_steps']
            self.agent.epsilon = checkpoint['epsilon']
            self.top_eval_score = checkpoint['top_eval_score']

    @log
    def evaluate(self, n_evals=5):
        eval_env = get_minenv(self.env_preferences)
        scores = 0
        wins = 0
        rewards = []
        for _ in range(n_evals):
            s, _ = eval_env.reset()
            done = False
            while not done:
                a = self.agent.act(s, training=False)
                s, reward, terminated, truncated, info = eval_env.step(a)
                done = terminated or truncated
            episode_score = info['score']
            if eval_env.unwrapped.game_won:
                wins += 1
            rewards.append(episode_score)
            scores += episode_score

        avg_return = np.round(scores / n_evals, 4)
        self.last_win_rate = wins / n_evals
        self.last_eval_scores = scores
        self.last_eval_rewards = rewards
        return avg_return

    @log
    def train(self):
        history = {'Step': [], 'AvgReturn': []}
        (s, _) = self.env.reset()
        self.start_time = int(time.time())
        writer = None
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(f'runs/{self.start_time}')
        while True:
            self.is_warmup = self.agent.warmup_steps > self.agent.total_steps
            if not self.is_warmup and self.non_warmup_start_time is None:
                self.non_warmup_start_time = int(time.time())
            a = self.agent.act(s)
            if isinstance(a, tp.SupportsInt):
                a = self.env.unwrapped.actions[a]
            self.last_action = a
            s_prime, r, terminated, truncated, info = self.env.step(a)
            result = self.agent.process((s, a, r, s_prime, terminated))
            self.last_reward = r
            if result:
                self.last_loss = result['value_loss']
                if writer:
                    writer.add_scalar('train/loss', result['value_loss'], result['total_steps'])
                    writer.add_scalar('train/epsilon', self.agent.epsilon, result['total_steps'])
            s = s_prime
            if terminated or truncated:
                s, _ = self.env.reset()

            if self.agent.total_steps % self.eval_interval == 0:
                self.evals_counter += 1
                ret = self.evaluate()
                self.last_avg_return = ret
                self.recent_eval_returns.append(ret)
                history['Step'].append(self.agent.total_steps)
                history['AvgReturn'].append(ret)
                if writer:
                    writer.add_scalar('eval/avg_return', ret, self.agent.total_steps)
                    writer.add_scalar('eval/win_rate', self.last_win_rate, self.agent.total_steps)
                if ret > self.top_eval_score:
                    self.top_eval_score = ret
                    if not self.is_warmup:
                        self.checkpoint_model()
                if self.evals_counter % 10 == 0:
                    self.create_history_plot(history)
            if self.env.render_mode == 'info':
                self.print_learning_data()
            if self.agent.total_steps > self.learning_max_steps:
                break
        self.create_history_plot(history)
        self.checkpoint_model(remove_previous=False)
        if writer:
            writer.close()

    def create_history_plot(self, history: dict[str, list]):
        plt.figure(figsize=(8, 5))
        plt.plot(history['Step'], history['AvgReturn'], 'r')
        plt.xlabel('Step', fontsize=16)
        plt.ylabel('AvgReturn', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis='y')
        plt.savefig(f'./evaluations/{self.agent.total_steps}plot.png')
        plt.close()
        with open('./evaluations/history.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'AvgReturn'])
            writer.writerows(zip(history['Step'], history['AvgReturn']))

    def print_learning_data(self):
        now = time.time()
        if now - self._last_print_time < 1.0:
            return
        self._last_print_time = now

        steps = self.agent.total_steps
        max_steps = self.learning_max_steps
        pct = steps / max(max_steps, 1)

        bar_len = 38
        filled = int(bar_len * pct)
        bar = '█' * filled + '░' * (bar_len - filled)

        elapsed = int(now - self.start_time)
        elapsed_str = str(datetime.timedelta(seconds=elapsed))
        if self.non_warmup_start_time and steps > self.agent.warmup_steps:
            secs_per_step = (now - self.non_warmup_start_time) / max(steps - self.agent.warmup_steps, 1)
            eta_secs = max(0, int(secs_per_step * (max_steps - steps)))
            eta_str = str(datetime.timedelta(seconds=eta_secs))
        else:
            eta_str = 'warming up...'

        trend = ''
        if len(self.recent_eval_returns) >= 2:
            delta = self.recent_eval_returns[-1] - self.recent_eval_returns[-2]
            trend = ' ↑' if delta > 0.5 else (' ↓' if delta < -0.5 else ' →')

        W = 48
        os.system('clear')
        print('═' * W)
        print('  MinesweeperAI — Training')
        print('═' * W)
        print(f'  [{bar}] {pct:.1%}')
        print(f'  Step:      {steps:>8,} / {max_steps:,}')
        print(f'  Elapsed:   {elapsed_str:<14}  ETA: {eta_str}')
        print()
        warmup_tag = '  [warmup]' if self.is_warmup else ''
        print(f'  Epsilon:   {self.agent.epsilon:.4f}{warmup_tag}')
        print(f'  Loss:      {self.last_loss:.4f}')
        print(f'  Reward:    {self.last_reward}')
        print()
        print(f'  {"── Evaluations ──":-<{W - 4}}')
        avg_str = f'{self.last_avg_return:.2f}' if self.recent_eval_returns else 'n/a'
        best_str = f'{self.top_eval_score:.2f}' if self.top_eval_score > -1e9 else 'n/a'
        print(f'  Avg return:  {avg_str}')
        print(f'  Win rate:    {self.last_win_rate:.1%}')
        print(f'  Best score:  {best_str}')
        if self.recent_eval_returns:
            history = '  '.join(f'{s:.1f}' for s in self.recent_eval_returns[-6:])
            print(f'  History:     {history}{trend}')
        print('═' * W)

    def checkpoint_model(self, remove_previous=True):
        if not self.is_warmup:
            if self.last_saved_model_name is not None and remove_previous:
                print('removing...', self.last_saved_model_name)
                os.remove(f'{MODELS_CHECKPOINTS_PATH}/{self.last_saved_model_name}')
            self.last_saved_model_name = (
                f'{self.start_time}_{self.agent.total_steps}_'
                f'{round(self.top_eval_score, 2)}_{self.model_filename}'
            )
            checkpoint = {
                'network': self.agent.network.state_dict(),
                'optimizer': self.agent.optimizer.state_dict(),
                'total_steps': self.agent.total_steps,
                'epsilon': self.agent.epsilon,
                'top_eval_score': self.top_eval_score,
            }
            torch.save(checkpoint, f'{MODELS_CHECKPOINTS_PATH}/{self.last_saved_model_name}')


def setup_teacher(teacher_kwargs: TeacherPreferences) -> Teacher:
    return Teacher(teacher_kwargs.agent_preferences,
                   teacher_kwargs.env_preferences,
                   teacher_kwargs.eval_interval,
                   teacher_kwargs.learning_max_steps,
                   teacher_kwargs.model_filename,
                   teacher_kwargs.resume_from,
                   teacher_kwargs.use_tensorboard)
