import gym
import gym.spaces
import pygame
from minesweeper_env.game.minesweeper_game import Minesweeper
import numpy as np

class MinesweeperEnv(Minesweeper, gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, field_size: tuple[int, int], mines_num: int, use_render: bool, seed: int | None = None, render_mode: str = 'human', max_steps: int = 200) -> None:
        super().__init__(field_size, mines_num, use_render, seed)
        self.window_size = 512
        self.action_space = gym.spaces.Discrete(field_size[0] * field_size[1] * 2)
        self.observation_space = gym.spaces.Box(-3., 8., (1, *field_size))
        self.render_mode = render_mode
        self.window = None
        self.actions = self._get_actions()
        self.lmc_actions = self._get_lmc_actions()
        self.lmc_actions_indices = [i for i in range(len(self.actions)) if i % 2 == 1]
        self.max_steps = max_steps
        self.render_mode = 'rgb_array'
        self.placed_flags = set()

        # self.step_ind = 1
        # self.reward = 0.
        # self.previous_player_field = None
        # self.previous_opened_field = None

    def _get_actions(self) -> list[list[int, int, int]]:
        actions = []
        for y in range(self.field_size[0]):
            for x in range(self.field_size[1]):
                actions.extend([[0, y, x], [1, y, x]])
        return actions

    def _get_lmc_actions(self) -> list[list[int, int, int]]:
        return [self.actions[i] for i in range(0,len(self.actions)) if i % 2 == 0]

    def reset(self, seed = None, options = None):
        self.previous_player_field = None
        self.previous_opened_field = None
        self.previous_action = None
        self.previous_good_flags = None
        self.reward = 0
        self.step_ind = 0
        self.reset_game()
        obs = np.asarray([self.opened_field])
        return obs, {}

    def step(self, action):
        self.previous_opened_field = np.copy(self.opened_field)
        self.previous_player_field = np.copy(self.player_field)
        self.previous_good_flags = self.placed_good_flags.copy()
        action = self.actions[action]
        self.step_ind += 1
        # print(action)
        self.previous_action = np.copy(action)
        self.play_action(action)
        obs = np.asarray([self.opened_field]) # add 1 to shape | (*field_size) ==> (1, *field_size)
        self._calculate_reward()
        return obs, self.reward, self.game_lost or self.game_won, self.step_ind > self.max_steps, {}

    def _calculate_reward(self):
        if self.game_lost:
            self.reward -= 20.
            return
        elif self.game_won:
            self.reward += 100.
            return
        if self.previous_action is not None:
            # If rightclick
            if self.previous_action[0] == 0.:
                # If good flag removed
                if len(self.placed_good_flags) < len(self.previous_good_flags):
                    self.reward -= 2.2
                # If rightclick was useless
                elif len(self.placed_good_flags) == len(self.previous_good_flags):
                    self.reward -= 0.2    
                # If good flag placed
                else:
                    self.reward += 2.



            elif self.previous_action[0] == 1.:
                # print('here')
                if self.previous_opened_field is not None:
                    # If clicked on opened cell
                    if (self.opened_field == self.previous_opened_field).all():
                        self.reward -= 0.01
                    # If clicked on closed cell and it wasnt mine
                    elif self.last_opened_cell_value != -3.:
                        self.reward += 0.5
                        # Bonus reward if clicked cell has opened neighbours
                        for nei in self.scanner.get_neighbours(self.last_opened_coords, self.field_size):
                            if self.opened_field[nei] >= 0.:
                                self.reward += 0.1

    def render(self):
        if self.render_mode == 'rgb_array':
            pass
        elif self.render_mode == 'human':
            self.human_render()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    

gym.register(id='MinesweeperEnv-v1',
             entry_point='minesweeper_env.minenv:MinesweeperEnv',
             max_episode_steps=300)

def get_minenv(field_size: int,
               mines_num: int,
               use_render: bool,
               seed: int | None = None,
               render_mode = 'human',
               max_steps = 200) -> MinesweeperEnv:
    return gym.make('MinesweeperEnv-v1',
                    field_size = field_size,
                    mines_num = mines_num,
                    use_render = use_render,
                    seed = seed,
                    render_mode = render_mode,
                    max_steps = max_steps)


