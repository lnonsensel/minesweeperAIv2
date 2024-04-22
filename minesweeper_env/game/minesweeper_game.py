import numpy as np
import random
from collections import deque
from minesweeper_env.game.generator import MinesweeperGenerator
from minesweeper_env.game.scanner import MinesweeperScanner
from minesweeper_env.game.renderer import MinesweeperRenderer

class Minesweeper:
    def __init__(self, field_size: tuple[int, int], mines_num: int, use_render: bool, seed: int | None = None) -> None:
        self.generator = MinesweeperGenerator()
        self.use_render = use_render
        self.field_size = field_size
        self.mines_num = mines_num
        self.placed_good_flags = set()
        self.seed = seed
        self.renderer = MinesweeperRenderer(field_size) if use_render else None
        self.reset_game()
        self.scanner = MinesweeperScanner(self.field)
        

    def reset_state(self):
        self.field = None
        self.opened_field = np.zeros(self.field_size, 'float32')
        self.opened_field[:, :] = -2.
        self.player_field = np.ones(self.field_size, 'float32')
        self.game_lost = False
        self.game_won = False
        self.last_opened_cell_value = None
        self.last_opened_coords = None
        self.utility_data = [{'Game lost': self.game_lost, 'Game won': self.game_won}]
        self.full_field = None
    
    def get_full_field(self):
        full_field = np.zeros(self.field_size)
        for y in range(self.field_size[0]):
            for x in range(self.field_size[1]):
                full_field[y, x] = self.scanner.get_cell_value((y, x))
        return full_field
        

    def reset_game(self):
        self.reset_state()
        self.human_render()

    def generate_fields(self, start_cell, seed: int | None = None):
        self.field = self.generator.generate_field((self.field_size, self.field_size),
                                                   start_cell,
                                                   self.mines_num,
                                                   self.seed)
    
    def check_game_end(self):
        if self.last_opened_cell_value == -3.:
            self.game_lost = True
            self.full_field[self.last_opened_coords] = -4.
        elif (self.player_field == self.field).all():
            self.game_won = True


    def _left_click_action(self, coords: tuple[int, int]):
        if self.field is None:
            self.field = self.generator.generate_field(self.field_size, coords, self.mines_num, self.seed)
            self.scanner.field = self.field
            self.full_field = self.get_full_field()
            print(self.full_field)
        if self.opened_field[coords] != -2:
            return
        cells_to_be_opened = [coords]
        clicked_cell_value = self.scanner.get_cell_value(coords)
        if clicked_cell_value == 0.:
            cells_to_be_opened.extend(self.scanner.get_neighbours_with_zero(coords))

        for cell in cells_to_be_opened:
            self.opened_field[cell[0]][cell[1]] = self.scanner.get_cell_value(cell)
            self.player_field[cell[0]][cell[1]] = 0. # if self.player_field[cell[0]][cell[1]] == 1. else 1.
        self.last_opened_cell_value = clicked_cell_value

    def _right_click_action(self, coords: tuple[int, int]):
        if self.field is None:
            return
        if self.opened_field[coords] != -2. and self.opened_field[coords] != -1.:
            return
        self.opened_field[coords] = -1. if self.opened_field[coords] == -2. else -2.
        print(self.full_field)
        if self.player_field[coords] == 1. and self.field[coords] == 1.:
            self.placed_good_flags.add(coords)
        elif coords in self.placed_good_flags:
            self.placed_good_flags.remove(coords)

    def play_action(self, action: tuple[int, int, int]):
        if self.game_lost or self.game_won:
            return
        is_left_click = action[0]
        coords = (action[1], action[2])
        self.last_opened_coords = coords
        if is_left_click:
            self._left_click_action(coords)
        else:
            self._right_click_action(coords)
        self.check_game_end()
        self.utility_data = [{'Game lost': self.game_lost, 'Game won': self.game_won}]
        self.human_render()

    def human_render(self):
        if self.renderer is None:
            return
        if self.game_lost or self.game_won:
            self.renderer.render(self.full_field, self.utility_data)
        else:
            self.renderer.render(self.opened_field, self.utility_data)
        self.renderer.update_display()