from minesweeper_env.game.ui import MinesweeperUI
from minesweeper_env.preferences import MinesweeperGamePreferences
from minesweeper_env.game.config import field_size, mines_num

def play_minesweeper(preferences: MinesweeperGamePreferences):
    ui = MinesweeperUI(preferences.field_size,
                       preferences.mines_num,
                       preferences.seed)
    while True:
        ui.run()

if __name__ == '__main__':
    play_minesweeper(MinesweeperGamePreferences(seed=None))