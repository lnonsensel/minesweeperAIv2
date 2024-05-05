from minesweeper_env.minenv import Minesweeper, MinesweeperEnv
from minesweeper_env.game.utils import MinesweeperUtils
import pygame as pg
import os

class MinesweeperUI(Minesweeper):
    def __init__(self, field_size: int, mines_num: int, seed: int | None = None) -> None:
        super().__init__(field_size, mines_num, True, seed)
    
    def quit(self):
        pg.quit()
        pg.display.quit()

    def get_action(self, button, y, x):
        y_c, x_c = self.renderer.utils.window_to_cell_coords((x, y))
        if button == 3:
            button = 0
        return [button, y_c, x_c]
    
    def run(self):
        for i in pg.event.get():
            if i.type == pg.MOUSEBUTTONDOWN:
                x, y = pg.mouse.get_pos()
                action = self.get_action(i.button, y, x)
                self.play_action(action)
                self.human_render()
            if i.type == pg.KEYDOWN:
                if i.key == pg.K_r:
                    self.reset_game()
            if i.type == pg.QUIT:
                self.quit()

def setup_ui(ui_kwargs) -> MinesweeperUI:
    return MinesweeperUI(**ui_kwargs)

class MinesweeperEnvUI(MinesweeperEnv):
    def __init__(self,
                 field_size: int,
                 mines_num: int,
                 seed: int | None = None) -> None:
        super().__init__(field_size, mines_num, True, seed, 'human', float('inf'))
        self.reset()

    def run(self):
        for i in pg.event.get():
            if i.type == pg.MOUSEBUTTONDOWN:
                x, y = pg.mouse.get_pos()
                x, y = self.renderer.utils.window_to_cell_coords((y, x))
                button = 1 if i.button == 1 else 0
                action = self.actions.index([button, y, x])
                obs, reward, terminated, truncated, info = self.step(action)
                # os.system('clear')
                print(reward)
                # print(action)
                # self.play_action(action)
            if i.type == pg.KEYDOWN:
                if i.key == pg.K_r:
                    self.reset()
            if i.type == pg.QUIT:
                self.close()
