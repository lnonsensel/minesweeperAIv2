from minesweeper_env.game.ui import MinesweeperUI
from minesweeper_env.game.config import field_size, mines_num

ui = MinesweeperUI(field_size, mines_num)
while True:
    ui.run()