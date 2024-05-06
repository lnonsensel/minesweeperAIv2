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
        self.agent_preferences.state_dim = self.env.observation_space.shape
        self.agent = get_agent(agent_preferences)
        self.top_eval_score = -float('inf')
        self.last_eval_scores = None
        self.last_eval_rewards = None
        self.max_reward = -float('inf')
        self.last_reward = None
        self.last_saved_model_name = None
        self.is_warmup = True
        self.non_warmup_start_time = None
        self.eval_interval = eval_interval
        self.learning_max_steps = learning_max_steps
        self.model_filename = model_filename

    def evaluate(self, n_evals=5):
        eval_env = get_minenv(self.env_preferences)
        scores = 0
        done = False
        rewards = []
        for _ in range(n_evals):
            s, _= eval_env.reset()
            s, reward, terminated, truncated, info = eval_env.step(eval_env.action_space.sample())
            while not done:
                a = self.agent.act(s, training=False)
                s, reward, terminated, truncated, info = eval_env.step(a)
                rewards.append(reward)
                done = terminated or truncated
            self.top_eval_score = reward if reward > self.top_eval_score else self.top_eval_score
            scores += reward
        
        self.last_eval_scores = scores
        self.last_eval_rewards = rewards
        return np.round(scores / n_evals, 4)

    def train(self):
        history = {'Step': [], 'AvgReturn': []}
        (s, _) = self.env.reset()

        s, reward, terminated, truncated, info = self.env.step(self.env.action_space.sample())
        self.start_time = int(time.time())
        
        while True:
            self.is_warmup = self.agent.warmup_steps > self.agent.total_steps
            if not self.is_warmup and self.non_warmup_start_time is None:
                self.non_warmup_start_time = int(time.time())
            a = self.agent.act(s)
            s_prime, r, terminated, truncated, info = self.env.step(a)
            result = self.agent.process((s, a, r, s_prime, terminated))
            self.last_reward = r

            
            s = s_prime
            if terminated or truncated:
                if self.max_reward < r:
                    self.max_reward = r
                    self.checkpoint_model()
                s, _ = self.env.reset()
                s, reward, terminated, truncated, info = self.env.step(self.env.action_space.sample())
                
            if self.agent.total_steps % self.eval_interval == 0:
                ret = self.evaluate()
                history['Step'].append(self.agent.total_steps)
                history['AvgReturn'].append(ret)
            
            # if self.env.render_mode in ['info', 'human']:
            #     self.print_learning_data(10)
            start_time = self.start_time
            non_warmup_start_time = self.non_warmup_start_time
            self.env.utility_data[0]['Round'] = f'{round(100 * (self.agent.total_steps / self.learning_max_steps), 2)}% | {self.agent.total_steps}/{self.learning_max_steps} | {int(time.time()) - start_time} seconds since start'
            if non_warmup_start_time is not None:
                avg_secs_step = (int(time.time()) - non_warmup_start_time) / self.agent.total_steps
                secs_left = avg_secs_step * (self.learning_max_steps - self.agent.total_steps)
                secs_left = str(datetime.timedelta(seconds=int(secs_left)))
            else:
                secs_left = 'Warmup ongoing'
            self.env.utility_data[0]['Seconds left'] = secs_left
            self.env.render()
            if self.agent.total_steps % self.eval_interval == 0:
                self.create_history_plot(history)
                self.checkpoint_model()
            if self.agent.total_steps > self.learning_max_steps:
                break
        self.create_history_plot(history)

    def create_history_plot(self, history: dict[str, int]):
        plt.figure(figsize=(8, 5))
        plt.plot(history['Step'], history['AvgReturn'], 'r')
        plt.xlabel('Step', fontsize=16)
        plt.ylabel('AvgReturn', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis='y')
        plt.savefig(f'./evaluations/{self.agent.total_steps}plot.png')

    def print_learning_data(self, update_rate = 1):
        if self.agent.total_steps % update_rate != 0:
            return
        os.system('clear')

    def save_learning_data(self, update_rate = 1):
        if self.agent.total_steps % update_rate != 0:
            return
        with open('./temp_data/data.txt', 'w') as file:
            for i in self.get_data():
                file.write(i)

    def get_data(self):       
        start_time = self.start_time
        non_warmup_start_time = self.non_warmup_start_time
        if non_warmup_start_time is not None:
            avg_secs_step = (int(time.time()) - non_warmup_start_time) / self.agent.total_steps
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
        top_eval_score = f'Top eval score: {self.top_eval_score}'
        return [percentage,
                steps_left,
                timer,
                time_left,
                points,
                warmup_status,
                eval_scores,
                top_eval_score]

    def checkpoint_model(self):
        if not self.is_warmup:
            if self.last_saved_model_name is not None:
                print('removing...', self.last_saved_model_name)
                os.remove(f'{MODELS_CHECKPOINTS_PATH}/{self.last_saved_model_name}')
            self.last_saved_model_name = f'{self.start_time}_{self.agent.total_steps}_{round(self.max_reward, 2)}_{self.model_filename}'
            torch.save(self.agent.network.state_dict(), f'{MODELS_CHECKPOINTS_PATH}/{self.last_saved_model_name}')


def setup_teacher(teacher_kwargs: TeacherPreferences) -> Teacher:
    
    return Teacher(teacher_kwargs.agent_preferences,
                   teacher_kwargs.env_preferences,
                   teacher_kwargs.eval_interval,
                   teacher_kwargs.learning_max_steps,
                   teacher_kwargs.model_filename)