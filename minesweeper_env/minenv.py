import gym
from gym import spaces
import pygame
from minesweeper_env.game.minesweeper_game import Minesweeper
import numpy as np
import os
from minesweeper_env.preferences import MinesweeperEnvPreferences
from dataclasses import asdict, dataclass

@dataclass
class LearningStepData:
    percentage: str
    steps_left: str
    timer: str
    time_left: str
    points: str
    warmup_status: str
    eval_scores: str
    top_eval_score: str

class MinesweeperEnv(Minesweeper, gym.Env):
    metadata = {"render_modes": ["human", "info", "none"]}
    def __init__(self,
                 field_size: tuple[int, int],
                 mines_num: int,
                 use_render: bool,
                 seed: int | None = None,
                 render_mode: str = 'human',
                 env_max_steps: int = 200,
                 **kwargs) -> None:
        use_render = use_render and render_mode == 'human'
        super().__init__(field_size, mines_num, use_render, seed)
        self.action_space = spaces.Discrete(self.field_size[0] * self.field_size[1])
        self.observation_space = spaces.Box(-4., 8, (1, *self.field_size))
        self.actions = self.get_actions()
        self.env_max_steps = env_max_steps
        self.render_mode = render_mode

    def step(self, action):
        self.previous_opened_field = np.copy(self.opened_field)
        self.previous_player_field = np.copy(self.player_field)
        self.previous_good_flags = self.placed_good_flags.copy()
        action = self.actions[action]
        self.play_action(action)
        obs = self.opened_field
        self.get_reward()
        self.render()
        self.step_ind += 1
        return obs, self.reward - self.last_reward, self.game_lost or self.game_won, self.step_ind > self.env_max_steps, {'score': self.reward}
    
    def reset(self, seed = None, options = None):
        self.previous_opened_field = None
        self.previous_player_field = None
        self.game_lost = False
        self.game_won = False
        self.reward = 0
        self.last_reward = 0
        self.step_ind = 0
        self.reset_game()
        return self.opened_field, {}
    
    def render(self):
        if self.render_mode == 'human':
            self.utility_data[0]['Points'] = self.reward
            self.utility_data[0]['Steps'] = f'{self.step_ind} / {self.env_max_steps}'
            self.utility_data[0]['Game won'] = self.game_won
            self.utility_data[0]['Game lost'] = self.game_lost
            self.human_render()
        # elif self.render_mode == 'info':
        #     os.system('clear')
        #     print(f'Steps: {self.step_ind} / {self.env_max_steps}')
        #     print(f'Score: {self.reward}')
        #     print(f'Game won: {self.game_won}')
        #     print(f'Game lost: {self.game_lost}')
        #     # print(*[i for i in asdict(info).items()], sep = '\n')
        elif self.render_mode == 'none':
            pass

    def close(self):
        if self.renderer is not None:
            pygame.display.quit()
            pygame.quit()

    def get_reward(self):
        self.last_reward = self.reward
        if self.game_lost:
            self.reward -= 0.
            return
        elif self.game_won:
            self.reward += 10.
            return
        if self.previous_opened_field is not None:
            # If clicked on opened cell
            if (self.opened_field == self.previous_opened_field).all():
                self.reward -= 0.1
            # If clicked on closed cell and it wasnt mine
            elif self.last_opened_cell_value != -3.:
                self.reward += 0.5
                # Bonus reward if clicked cell has opened neighbours
                for nei in self.scanner.get_neighbours(self.last_opened_coords, self.field_size):
                    if self.opened_field[nei] >= 0.:
                        self.reward += 0.1

    def get_actions(self):
        actions = []
        for y in range(self.field_size[0]):
            for x in range(self.field_size[1]):
                actions.append((1, y, x,))
        return actions
    
gym.register(id='MinesweeperEnv-v1',
             entry_point='minesweeper_env.minenv:MinesweeperEnv',
             max_episode_steps=300)

def get_minenv(preferences: MinesweeperEnvPreferences | dict) -> MinesweeperEnv:
    if isinstance(preferences, MinesweeperEnvPreferences):
        return gym.make('MinesweeperEnv-v1',
                        **asdict(preferences.game_preferences),
                        **asdict(preferences))
    elif isinstance(preferences, dict):
        return gym.make('MinesweeperEnv-v1',
                        **preferences['game_preferences'], **preferences)

