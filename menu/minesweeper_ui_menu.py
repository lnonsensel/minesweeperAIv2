from minesweeper_env.preferences import MinesweeperGamePreferences
from menu.elements import TkinterElements
import typing as tp

class MinesweeperUIMenu:
    def __init__(self, tab_control, start_command: tp.Callable = lambda x: x) -> None:
        self.ui_preferences = MinesweeperGamePreferences()
        self.ui_elements = TkinterElements(tab_control, 'UI')
        self.start_command = start_command
        self.use_ui = False

    def add_start_button(self, grid: tuple[int, int]):
        self.ui_elements.add_button(grid, 'Start UI', self.execute_start_command)

    def add_use_seed_checker(self, grid: tuple[int, int]):
        self.use_seed_checker_ui_data = self.ui_elements.add_checker(grid, 'Use seed', 'Seed')

    def add_seed_spinbox(self, grid: tuple[int, int]):
        self.seed_spinbox_ui_data = self.ui_elements.add_spinbox(grid, 'Seed', self.ui_preferences.seed, increment=1)

    def add_mines_num_spinbox(self, grid: tuple[int, int]):
        self.mines_num_spinbox_ui_data = self.ui_elements.add_spinbox(grid, 'Mines number', self.ui_preferences.mines_num, increment=10)

    def add_field_size_spinbox(self, grid: tuple[int, int]):
        self.field_size_spinbox_ui_data = self.ui_elements.add_spinbox(grid, 'Field side size', self.ui_preferences.field_size[0], increment=10)

    def execute_start_command(self):
        self.update_preferences()
        self.use_ui = True
        self.ui_elements.tab_control.quit()
        self.ui_elements.tab_control.destroy()
        self.start_command(self.ui_preferences)

    def update_ui(self):
        if not self.use_seed_checker_ui_data.variable.get():
            self.seed_spinbox_ui_data.element['state'] = 'disabled'
        else:
            self.seed_spinbox_ui_data.element['state'] = 'normal'

    def update_preferences(self):
        self.ui_preferences.mines_num = self.mines_num_spinbox_ui_data.variable.get()
        field_size = (self.field_size_spinbox_ui_data.variable.get(), self.field_size_spinbox_ui_data.variable.get(),)
        self.ui_preferences.field_size = field_size
        self.ui_preferences.seed = self.seed_spinbox_ui_data.variable.get() if self.use_seed_checker_ui_data.variable.get() else None
    
    def add_ui_to_window(self):
        elements = [self.add_mines_num_spinbox,
                    self.add_field_size_spinbox,
                    self.add_use_seed_checker,
                    self.add_seed_spinbox,
                    self.add_start_button]
        self.ui_elements.pack_tab(elements)