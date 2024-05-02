from tkinter import *
from tkinter.ttk import Notebook
from menu.minesweeper_env_menu import MinesweeperEnvMenu
from menu.minesweeper_ui_menu import MinesweeperUIMenu
from menu.teacher_menu import TeacherMenu
from menu.agent_menu import AgentMenu
from menu.menu_config import menu_window_name, menu_window_size
import typing as tp

class MinesweeperAIMenu:
    def __init__(self, env_entry_point: tp.Callable, ui_entry_point: tp.Callable) -> None:
        self.window = Tk()
        self.window.title = menu_window_name
        self.window.geometry = menu_window_size
        self.tab_control = Notebook(self.window)
        self.env_menu = MinesweeperEnvMenu(self.tab_control)
        self.ui_menu = MinesweeperUIMenu(self.tab_control)
        self.teacher_menu = TeacherMenu(self.tab_control)
        self.agent_menu = AgentMenu(self.tab_control)

    def init_menus(self):
        self.env_menu.add_env_to_window()
        self.ui_menu.add_ui_to_window()
        self.teacher_menu.add_teacher_to_window()
        self.agent_menu.add_agent_to_menu()
        self.tab_control.pack()

    def update(self):
        self.env_menu.update_env()
        self.ui_menu.update_ui()
        self.window.after(100, self.update)

    def run(self):
        self.init_menus()
        self.update()
        self.window.mainloop()


if __name__ == '__main__':
    def f():
        print('hello!')
    menu = MinesweeperAIMenu(f)
    menu.run()