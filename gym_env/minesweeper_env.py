from gym import Env
from game.minesweeper_game import Minesweeper

class MinesweeperEnv(Env):
    def __init__(self, game_instance: Minesweeper):
        self.game = game_instance