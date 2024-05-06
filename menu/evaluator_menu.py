from tkinter.ttk import Notebook
from menu.elements import TkinterElements
from teacher.config import MODELS_CHECKPOINTS_PATH
from teacher.preferences import EvaluatorPreferences
import os
class EvaluatorMenu:
    def __init__(self, tab_control) -> None:
        # self.eval_preferences = EvaluatorPreferences()
        self.eval_elements = TkinterElements(tab_control, 'Evaluator')

    def add_model_filename_box(self, grid: tuple[int, int]):
        model_filenames = [i for i in os.listdir(MODELS_CHECKPOINTS_PATH) if i.endswith('.pt')]
        self.eval_elements.add_box(grid, 'Model filename', model_filenames, model_filenames[-1])
    
    def add_start_eval_button(self, grid: tuple[int, int]):
        self.eval_elements.add_button(grid, 'Start Eval', self.execute_start_command)

    def execute_start_command(self):
        self.eval_elements.tab_control.quit()
        self.eval_elements.tab_control.destroy()

    def add_eval_to_window(self):
        elements = [
            self.add_model_filename_box,
            self.add_start_eval_button
        ]
        self.eval_elements.pack_tab(elements)
