from menu.elements import TkinterElements
from teacher.preferences import TeacherPreferences

class TeacherMenu:
    def __init__(self, tab_control) -> None:
        self.teacher_elements = TkinterElements(tab_control, 'Teacher')
        self.teacher_preferences = TeacherPreferences()

    def add_eval_interval_spinbox(self, grid: tuple[int, int]):
        self.eval_interval_spinbox_data = self.teacher_elements.add_spinbox(grid, 'Eval interval', self.teacher_preferences.eval_interval, increment=1000)

    def add_model_filename_input(self, grid: tuple[int, int]):
        self.model_filename_input_data = self.teacher_elements.add_input(grid, 'Model filename', self.teacher_preferences.model_filename)

    def add_max_learning_steps_input(self, grid: tuple[int, int]):
        self.max_learning_steps_input = self.teacher_elements.add_input(grid, 'Learning max steps', self.teacher_preferences.learning_max_steps)

    def update_preferences(self):
        self.teacher_preferences.eval_interval = self.eval_interval_spinbox_data.variable.get()
        self.teacher_preferences.model_filename = self.model_filename_input_data.variable.get()
        self.teacher_preferences.learning_max_steps = self.max_learning_steps_input.variable.get()

    def add_teacher_to_window(self):
        elements = [self.add_eval_interval_spinbox,
                    self.add_model_filename_input,
                    self.add_max_learning_steps_input]
        self.teacher_elements.pack_tab(elements)
