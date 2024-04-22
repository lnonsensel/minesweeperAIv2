import pygame as pg
from minesweeper_sprites.minesweeper import sprites
from minesweeper_env.game.utils import MinesweeperUtils
from minesweeper_env.game.renderer_config import window_name, window_size, field_area_size, utility_area_size
import numpy as np
import pygame
import torch

class MinesweeperSprites:
    def get_sprites(self):
        two_thousand_tiles = sprites.TileSheets(sprites.TileSheets.two_thousand)
        tiles_builder = sprites.TileBuilder(two_thousand_tiles)
        tile = tiles_builder.build()

        two_thousand_scores = sprites.ScoreSheets(sprites.ScoreSheets.two_thousand)
        scores_builder = sprites.ScoreBuilder(two_thousand_scores)
        scores = scores_builder.build()
        return tile, scores
        
class MinesweeperRenderer:
    def __init__(self, field_size):
        pygame.init()
        pygame.display.set_caption(window_name)
        self.sc = pygame.display.set_mode(window_size)
        self.field_size = field_size
        self.window_size = window_size
        self.field_area_size = field_area_size
        self.utility_area_size = utility_area_size
        self.cell_size = self.field_area_size[0] // field_size[0]
        self.utils = MinesweeperUtils(self.cell_size, self.utility_area_size, self.field_area_size)
        self.tile, self.scores = MinesweeperSprites().get_sprites()

    def update_display(self):
        pygame.display.update()

    def _get_rect(self, cell_coords: tuple[int, int]):
        window_coords = self.utils.cell_to_window_coords(cell_coords)
        return pygame.rect.Rect(*window_coords, self.cell_size, self.cell_size)

    def draw_transparent_rect(self, cell_coords: tuple[int, int], color: int, transparency: int = 127):
        rect = self._get_rect(cell_coords)
        shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, (*color, transparency), shape_surf.get_rect())
        self.sc.blit(shape_surf, rect)

    def draw_utility_data(self, data: dict, column_shift: int = 0):
        bg = self.tile.flag
        pygame.draw.rect(self.sc, (100, 100, 100), (0, 0, *self.utility_area_size))
        rows = [f'{k}: {v}' for k, v in data.items()]
        for row, coord in zip(rows, range(0, utility_area_size[1], 15)):
            self.put_text(row, 30, (column_shift, coord), (255, 0, 0))

    def draw_img_in_cell(self, image_number: float, cell_coords: tuple[int, int]):
        window_coords = self.utils.cell_to_window_coords(cell_coords)
        image_number = int(image_number)
        if image_number == -3:
            img = self.tile.mine
        elif image_number == -2:
            img = self.tile.unopened
        elif image_number == -1:
            img = self.tile.flag
        elif image_number == -4:
            img = self.tile.mine_red_cross
        else:
            img = self.tile[image_number]
        img = pygame.transform.scale(img, (self.cell_size, self.cell_size))
        self.sc.blit(img, window_coords)

    
    def put_text(self, text: str, font_size: int = 72, coords: tuple[int, int] = (10, 10), color: tuple[int, int, int] = (255, 0, 0)):
        font = pg.font.SysFont('Cour.ttf', font_size)
        text_img = font.render(text, True, color)
        self.sc.blit(text_img, coords)

    
    def render_opened_field(self, opened_field: np.ndarray):
        for y in range(self.field_size[0]):
            for x in range(self.field_size[1]):
                self.draw_img_in_cell(opened_field[y][x], (y, x))

    
    def render_q_weights(self, q_weights_field: np.ndarray, color: tuple[int, int, int], top_index: int = 5):
        q = torch.clone(q_weights_field)
        top_values = torch.sort(q.reshape((np.prod(self.field_size),))).values[-top_index::]
        for y in range(q_weights_field.shape[0]):
            for x in range(q_weights_field.shape[1]):
                
                value = q_weights_field[y, x]
                if value not in top_values:
                    value = 0
                else:
                    for i in range(len(top_values)):
                        if top_values[i] == value:
                            value = i / len(top_values)
                
                if value != 0:
                    value = value.item() if isinstance(value, torch.Tensor) else value
                transparency_coeff = int(255 * value)
                self.draw_transparent_rect((y, x), color, transparency_coeff)
    
    def game_end_render_field(self, field: np.ndarray):
        pass

    def render(self, opened_field: np.ndarray, data_to_render: list[dict] = []):
        for data, shift in zip(data_to_render, range(0, 500, 50)):
            self.draw_utility_data(data, shift)
        self.render_opened_field(opened_field)