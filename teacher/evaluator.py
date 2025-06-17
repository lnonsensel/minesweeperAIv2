from teacher.config import MODELS_CHECKPOINTS_PATH
import torch
import gymnasium as gym
import random
import os
from agent.dqn import DQN
from minesweeper_env.minenv import get_minenv, MinesweeperEnvPreferences
from teacher.preferences import EvaluatorPreferences
import time

class Evaluator:
    def __init__(self, env_preferences: MinesweeperEnvPreferences, eval_preferences: EvaluatorPreferences) -> None:
        self.env: gym.Env = get_minenv(env_preferences)
        self.model_filename = eval_preferences.model_filename

    def get_loaded_agent(self):
        dqn = DQN(self.env.observation_space.shape, self.env.action_space.n)
        dqn.network.load_state_dict(torch.load(f'{MODELS_CHECKPOINTS_PATH}/{self.model_filename}'))
        return dqn

    def collect_eval_frames(self):
        eval_env = self.env
        eval_agent = self.get_loaded_agent()
        scores = 0
        s, _ = eval_env.reset()
        done = False
        s_prime, r, terminated, truncated, info = eval_env.step(random.randint(0, eval_env.action_space.n))
        s = s_prime
        while True:
            eval_env.render()
            time.sleep(0.5)
            a = eval_agent.act(s, training=False)
            # print(a)
            s_prime, r, terminated, truncated, info = eval_env.step(a)
            s = s_prime
            done = terminated or truncated
            
            os.system('clear')
            print(a)
            print(f'Reward: {r}\nDone: {done}')

def get_evaluator(env_preferences: MinesweeperEnvPreferences, eval_preferences: EvaluatorPreferences):
    return Evaluator(env_preferences, eval_preferences)

def start_evaluator(env_preferences: MinesweeperEnvPreferences, eval_preferences: EvaluatorPreferences):
    print(env_preferences, eval_preferences)
    evaluator = get_evaluator(env_preferences, eval_preferences)
    evaluator.collect_eval_frames()