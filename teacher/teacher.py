import csv
from agent.dqn import DQN
from minesweeper_env.minenv import get_minenv, LearningStepData
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
from dataclasses import dataclass, asdict
import typing as tp
from utils import log

class Teacher:
    def __init__(self,
                 agent_preferences: AgentPreferences,
                 env_preferences: MinesweeperEnvPreferences,
                 eval_interval: int,
                 learning_max_steps: int,
                 model_filename: str) -> None:
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
            if result and self.env.render_mode == 'info':
                print(
                    f"step={result['total_steps']} "
                    f"loss={result['value_loss']:.4f} "
                    f"eps={self.agent.epsilon:.4f}"
                )
            s = s_prime
            if terminated or truncated:
                s, _ = self.env.reset()

            if self.agent.total_steps % self.eval_interval == 0:
                self.evals_counter += 1
                ret = self.evaluate()
                history['Step'].append(self.agent.total_steps)
                history['AvgReturn'].append(ret)
                if ret > self.top_eval_score:
                    self.top_eval_score = ret
                    if not self.is_warmup:
                        self.checkpoint_model()
                if self.evals_counter % 10 == 0:
                    self.create_history_plot(history)
            if self.env.render_mode == 'info':
                self.print_learning_data(5)
            if self.agent.total_steps > self.learning_max_steps:
                break
        self.create_history_plot(history)
        self.checkpoint_model(remove_previous=False)

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

    def print_learning_data(self, update_rate=1):
        if self.agent.total_steps % update_rate != 0:
            return
        os.system('clear')
        learning_data = self.get_data()
        print(*[i for i in asdict(learning_data).values()], sep='\n')
        print(self.last_action)

    def save_learning_data(self, update_rate=1):
        if self.agent.total_steps % update_rate != 0:
            return
        with open('./temp_data/data.txt', 'w') as file:
            for i in self.get_data():
                file.write(i)

    @log
    def get_data(self):
        start_time = self.start_time
        non_warmup_start_time = self.non_warmup_start_time
        if non_warmup_start_time is not None:
            avg_secs_step = (int(time.time()) - non_warmup_start_time) / max(self.agent.total_steps, 1)
            secs_left = avg_secs_step * (self.learning_max_steps - self.agent.total_steps)
            secs_left = str(datetime.timedelta(seconds=int(secs_left)))
        else:
            secs_left = 'Warmup ongoing'
        percentage = f'{round(100 * (self.agent.total_steps / self.learning_max_steps), 2)}%'
        steps_left = f'{self.agent.total_steps}/{self.learning_max_steps}'
        timer = f'{int(time.time()) - start_time} seconds since start'
        time_left = f'~Time left: {secs_left}'
        points = f'Points: {self.last_reward}'
        warmup_status = f'Warmup: {self.is_warmup}'
        eval_scores = f'Last eval scores: {self.last_eval_scores}'
        top_eval_score = f'Top eval score: {self.top_eval_score} | Win rate: {self.last_win_rate:.1%}'
        return LearningStepData(percentage, steps_left, timer, time_left, points, warmup_status, eval_scores, top_eval_score)

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
                   teacher_kwargs.model_filename)
