import gymnasium as gym
from gymnasium import spaces
import pygame
from minesweeper_env.game.minesweeper_game import Minesweeper
from minesweeper_env.game.config import RewardConfig
import numpy as np
import os
from minesweeper_env.preferences import MinesweeperEnvPreferences
from dataclasses import asdict
import typing as tp

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
        if mines_num >= field_size[0] * field_size[1]:
            raise ValueError(
                f"mines_num ({mines_num}) must be less than total cells "
                f"({field_size[0] * field_size[1]})"
            )
        use_render = use_render and render_mode == 'human'
        super().__init__(field_size, mines_num, use_render, seed)
        self.action_space = spaces.Discrete(self.field_size[0] * self.field_size[1] * 2)
        self.observation_space = spaces.Box(-4., 8, (1, *self.field_size))
        self.actions = self.get_actions()
        self.env_max_steps = env_max_steps
        self.render_mode = render_mode
        self.rc = RewardConfig()

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
        step_reward = self.reward - self.last_reward
        if self.rc.reward_clip > 0.0:
            step_reward = float(np.clip(step_reward, -self.rc.reward_clip, self.rc.reward_clip))
        return obs, step_reward, self.game_lost or self.game_won, self.step_ind > self.env_max_steps, {'score': self.reward}

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
            pass  # display is managed by Teacher.print_learning_data
        elif self.render_mode == 'none':
            pass

    def close(self):
        if self.renderer is not None:
            pygame.display.quit()
            pygame.quit()

    def get_reward(self):
        self.last_reward = self.reward
        step_reward = 0
        rc = self.rc

        if self.game_lost:
            step_reward += rc.loss_penalty
            self.reward += step_reward
            return

        if self.game_won:
            win_bonus = max(rc.win_min, rc.win_base - self.step_ind * rc.win_speed_factor)
            step_reward += win_bonus
            self.reward += step_reward
            return

        step_reward += rc.step_penalty

        if self.previous_opened_field is not None:
            if (self.opened_field == self.previous_opened_field).all():
                step_reward += rc.repeat_click_penalty
            elif self.last_opened_cell_value != -3:
                step_reward += rc.safe_cell_reward

                if self.last_opened_cell_value > 0:
                    step_reward += self.last_opened_cell_value * rc.neighbor_info_factor

                open_neighbors = 0
                for nei in self.scanner.get_neighbours(self.last_opened_coords, self.field_size):
                    if self.opened_field[nei] >= 0.:
                        open_neighbors += 1
                        step_reward += rc.open_neighbor_reward

                if open_neighbors >= rc.surround_threshold:
                    step_reward += rc.surround_bonus

        # Rewards for flag placement (raw field: mines == 1., safe == 0.)
        if self.last_action_type == 0 and self.last_action_coords is not None:
            y, x = self.last_action_coords
            if self.opened_field[y, x] == -1.:  # flag was just placed, not removed
                cell_value = self.field[y, x]
                if cell_value == 1.:  # mine
                    step_reward += rc.correct_flag_reward
                    mine_neighbors = sum(
                        1 for nei in self.scanner.get_neighbours((y, x), self.field_size)
                        if self.field[nei[0], nei[1]] == 1.
                    )
                    step_reward += mine_neighbors * rc.mine_cluster_factor
                elif cell_value == 0.:  # safe cell — wrong flag
                    step_reward += rc.wrong_flag_penalty

        if self.previous_opened_field is not None:
            new_opened = np.sum(self.opened_field != -2) - np.sum(self.previous_opened_field != -2)
            step_reward += new_opened * rc.new_cell_factor

            correct_flags = np.sum((self.opened_field == -1.) & (self.field == 1.))
            prev_correct = len(self.previous_good_flags) if self.previous_good_flags is not None else 0
            new_correct_flags = correct_flags - prev_correct
            step_reward += new_correct_flags * rc.correct_flag_factor

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
