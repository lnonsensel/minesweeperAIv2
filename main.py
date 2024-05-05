from menu.menu import MinesweeperAIMenu
from minesweeper_env.game.ui import setup_ui
from dataclasses import asdict
from teacher.teacher import setup_teacher
import time

menu = MinesweeperAIMenu()
menu.run()
menu.window.quit()
menu.window.destroy()
time.sleep(1)
if menu.menu_data.use_ui:
    ui = setup_ui(asdict(menu.menu_data.ui_data))
    while True:
        ui.run()
else:
    teacher = setup_teacher(menu.menu_data.teacher_data)
    teacher.train()