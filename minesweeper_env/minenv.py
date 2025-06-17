import gymnasium as gym
from gymnasium import spaces
import pygame
from minesweeper_env.game.minesweeper_game import Minesweeper
import numpy as np
import os
from minesweeper_env.preferences import MinesweeperEnvPreferences
from dataclasses import asdict, dataclass
import typing as tp

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
        self.action_space = spaces.Discrete(self.field_size[0] * self.field_size[1] * 2)
        self.observation_space = spaces.Box(-4., 8, (1, *self.field_size))
        self.actions = self.get_actions()
        self.env_max_steps = env_max_steps
        self.render_mode = render_mode

    def step(self, action):
        self.previous_opened_field = np.copy(self.opened_field)
        self.previous_player_field = np.copy(self.player_field)
        self.previous_good_flags = self.placed_good_flags.copy()
        if isinstance(action, tp.SupportsInt):
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
        elif self.render_mode == 'info':
            os.system('clear')
            print(f'Steps: {self.step_ind} / {self.env_max_steps}')
            print(f'Score: {self.reward}')
            print(f'Game won: {self.game_won}')
            print(f'Game lost: {self.game_lost}')
            # print(*[i for i in asdict(info).items()], sep = '\n')
        elif self.render_mode == 'none':
            pass

    def close(self):
        if self.renderer is not None:
            pygame.display.quit()
            pygame.quit()

    # def get_reward(self):
    #     self.last_reward = self.reward
    #     if self.game_lost:
    #         self.reward -= 0.
    #         return
    #     elif self.game_won:
    #         self.reward += 10.
    #         return
    #     if self.previous_opened_field is not None:
    #         # If clicked on opened cell
    #         if (self.opened_field == self.previous_opened_field).all():
    #             self.reward -= 0.1
    #         # If clicked on closed cell and it wasnt mine
    #         elif self.last_opened_cell_value != -3.:
    #             self.reward += 0.5
    #             # Bonus reward if clicked cell has opened neighbours
    #             for nei in self.scanner.get_neighbours(self.last_opened_coords, self.field_size):
    #                 if self.opened_field[nei] >= 0.:
    #                     self.reward += 0.1
    def get_reward(self):
        self.last_reward = self.reward
        step_reward = 0
        
        # 1. Крупные награды за победу/поражение
        if self.game_lost:
            step_reward -= 50  # Значительный штраф за поражение
            self.reward += step_reward
            return
            
        if self.game_won:
            # Награда за победу + бонус за скорость
            win_bonus = max(0, 100 - self.step_ind * 0.5)
            step_reward += win_bonus
            self.reward += step_reward
            return
        
        # 2. Штраф за каждый ход (стимулирует эффективность)
        step_reward -= 0.3
        
        # 3. Награды за открытие клеток
        if self.previous_opened_field is not None:
            # Штраф за повторное открытие
            if (self.opened_field == self.previous_opened_field).all():
                step_reward -= 2.0  # Увеличенный штраф
            
            # Награда за новую безопасную клетку
            elif self.last_opened_cell_value != -3:
                # Базовая награда
                step_reward += 1.5
                
                # Бонус за информативность (чем больше мин вокруг, тем выше награда)
                if self.last_opened_cell_value > 0:
                    step_reward += self.last_opened_cell_value * 0.8
                
                # Бонус за открытие рядом с открытыми областями
                open_neighbors = 0
                for nei in self.scanner.get_neighbours(self.last_opened_coords, self.field_size):
                    if self.opened_field[nei] >= 0.:
                        open_neighbors += 1
                        step_reward += 0.3
                
                # Супербонус за полное окружение
                if open_neighbors >= 4:
                    step_reward += 2.0
        
        # 4. Награды за установку флагов
        if hasattr(self, 'last_action_type') and self.last_action_type == 0:  # Flag action
            y, x = self.last_action_coords
            cell_value = self.field[y, x]
            
            # Правильная установка флага на мину
            if cell_value == -3:
                step_reward += 5.0
                # Дополнительный бонус за обнаружение кластера мин
                mine_neighbors = sum(1 for nei in self.scanner.get_neighbours((y, x), self.field_size) 
                                if self.field[nei[0], nei[1]] == -3)
                step_reward += mine_neighbors * 0.7
            
            # Неправильная установка флага
            elif cell_value >= 0:
                step_reward -= 4.0
        
        # 5. Стратегические бонусы
        # Бонус за уменьшение неопределенности
        new_opened = np.sum(self.opened_field != -2) - np.sum(self.previous_opened_field != -2)
        step_reward += new_opened * 0.4
        
        # Бонус за продвижение к победе
        correct_flags = np.sum((self.player_field == -1) & (self.field == -3))
        step_reward += correct_flags * 0.2
        
        self.reward += step_reward
    def get_actions(self):
        actions = []
        for y in range(self.field_size[0]):
            for x in range(self.field_size[1]):
                actions.append((1, y, x,))
        for y in range(self.field_size[0]):
            for x in range(self.field_size[1]):
                actions.append((0, y, x,))
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

