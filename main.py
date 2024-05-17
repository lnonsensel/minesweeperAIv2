from menu.menu import MinesweeperAIMenu
from minesweeper_env.game.ui import setup_ui
from dataclasses import asdict
from teacher.teacher import setup_teacher
from teacher.evaluator import start_evaluator
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
elif menu.menu_data.use_env:
    teacher = setup_teacher(menu.menu_data.teacher_data)
    teacher.train()
elif menu.menu_data.use_evaluator:
    start_evaluator(menu.menu_data.env_data, menu.menu_data.eval_data)