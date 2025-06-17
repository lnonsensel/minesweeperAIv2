from menu.menu import MinesweeperAIMenu
import time

menu = MinesweeperAIMenu()
menu.run()
menu.window.quit()
menu.window.destroy()
time.sleep(0.1)
if menu.menu_data.use_ui:
    from minesweeper_env.game.ui import setup_ui
    from dataclasses import asdict
    ui = setup_ui(asdict(menu.menu_data.ui_data))
    while True:
        ui.run()
elif menu.menu_data.use_env:
    from teacher.teacher import setup_teacher
    teacher = setup_teacher(menu.menu_data.teacher_data)
    teacher.train()
elif menu.menu_data.use_evaluator:
    from teacher.evaluator import start_evaluator
    start_evaluator(menu.menu_data.env_data, menu.menu_data.eval_data)