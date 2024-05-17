from tkinter.ttk import Notebook
from menu.elements import TkinterElements
from teacher.config import MODELS_CHECKPOINTS_PATH
from teacher.preferences import EvaluatorPreferences
import os
class EvaluatorMenu:
    def __init__(self, tab_control) -> None:
        self.eval_preferences = EvaluatorPreferences('none')
        self.eval_elements = TkinterElements(tab_control, 'Evaluator')
        self.use_evaluator = False

    def add_model_filename_box(self, grid: tuple[int, int]):
        model_filenames = [i for i in sorted(os.listdir(MODELS_CHECKPOINTS_PATH)) if i.endswith('.pt')]
        if len(model_filenames) == 0:
            model_filenames.append('none')
        default_model_filename = model_filenames[-1]
        self.model_filename_box = self.eval_elements.add_box(grid, 'Model filename', model_filenames, default_model_filename)
    
    def add_start_eval_button(self, grid: tuple[int, int]):
        self.start_eval_button = self.eval_elements.add_button(grid, 'Start Eval', self.execute_start_command)

    def execute_start_command(self):
        self.eval_elements.tab_control.quit()
        self.eval_elements.tab_control.destroy()
        self.use_evaluator = True

    def update_preferences(self):
        self.eval_preferences.model_filename = self.model_filename_box.element.get()

    def add_eval_to_window(self):
        elements = [
            self.add_model_filename_box,
            self.add_start_eval_button
        ]
        self.eval_elements.pack_tab(elements)
