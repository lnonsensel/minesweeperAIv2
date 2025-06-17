from tkinter import *
from tkinter.ttk import Notebook
from menu.minesweeper_env_menu import MinesweeperEnvMenu
from menu.minesweeper_ui_menu import MinesweeperUIMenu
from menu.teacher_menu import TeacherMenu
from menu.agent_menu import AgentMenu
from menu.menu_config import menu_window_name, menu_window_size
import typing as tp
from teacher.preferences import TeacherPreferences, EvaluatorPreferences
from minesweeper_env.preferences import MinesweeperGamePreferences, MinesweeperEnvPreferences
from dataclasses import dataclass
from menu.evaluator_menu import EvaluatorMenu

@dataclass
class MenuFinalData:
    use_ui: bool
    use_env: bool
    use_evaluator: bool
    env_data: MinesweeperEnvPreferences
    teacher_data: TeacherPreferences
    ui_data: MinesweeperGamePreferences
    eval_data: EvaluatorPreferences

class MinesweeperAIMenu:
    def __init__(self) -> None:
        self.window = Tk()
        self.window.title = menu_window_name
        self.window.geometry = menu_window_size
        self.tab_control = Notebook(self.window)
        self.env_menu = MinesweeperEnvMenu(self.tab_control)
        self.ui_menu = MinesweeperUIMenu(self.tab_control)
        self.teacher_menu = TeacherMenu(self.tab_control)
        self.agent_menu = AgentMenu(self.tab_control)
        self.eval_menu = EvaluatorMenu(self.tab_control)
        self.menu_data = MenuFinalData(False, False, False, MinesweeperEnvPreferences(), TeacherPreferences(), MinesweeperGamePreferences(), EvaluatorPreferences())

    def init_menus(self):
        self.env_menu.add_env_to_window()
        self.ui_menu.add_ui_to_window()
        self.teacher_menu.add_teacher_to_window()
        self.agent_menu.add_agent_to_menu()
        self.eval_menu.add_eval_to_window()
        self.tab_control.pack()

    def update_menu_data(self):
        self.menu_data.use_ui = self.ui_menu.use_ui
        self.menu_data.use_env = self.env_menu.use_env
        self.menu_data.use_evaluator = self.eval_menu.use_evaluator
        self.teacher_menu.teacher_preferences.env_preferences = self.env_menu.env_preferences
        self.teacher_menu.teacher_preferences.agent_preferences = self.agent_menu.agent_preferences
        self.menu_data.env_data = self.env_menu.env_preferences
        self.menu_data.teacher_data = self.teacher_menu.teacher_preferences
        self.menu_data.ui_data = self.ui_menu.ui_preferences
        self.menu_data.eval_data = self.eval_menu.eval_preferences

    def update_all_preferences(self):
        self.env_menu.update_env_preferences()
        self.ui_menu.update_preferences()
        self.teacher_menu.update_preferences()
        self.agent_menu.update_preferences()
        self.eval_menu.update_preferences()

    def update(self):
        self.update_menu_data()
        self.update_all_preferences()
        self.env_menu.update_env()
        self.ui_menu.update_ui()
        self.window.after(100, self.update)

    def run(self):
        self.init_menus()
        self.update()
        self.window.mainloop()
        self.update_menu_data()