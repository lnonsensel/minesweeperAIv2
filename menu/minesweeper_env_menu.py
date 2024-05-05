from minesweeper_env.preferences import MinesweeperEnvPreferences
from menu.elements import TkinterElements
import typing as tp
class MinesweeperEnvMenu:
    def __init__(self, tab_control, start_command: tp.Callable = lambda x: x) -> None:
        self.env_preferences = MinesweeperEnvPreferences()
        self.env_elements = TkinterElements(tab_control, 'Environment')
        self.start_command = start_command

    def add_render_mode_box(self, grid: tuple[int, int]):
        self.render_mode_box_data = self.env_elements.add_box(grid, 'Render mode', self.env_preferences.render_modes, self.env_preferences.render_mode)

    def add_use_render_checker(self, grid: tuple[int, int]):
        self.use_render_checker_data = self.env_elements.add_checker(grid, 'Use Render', 'Render')

    def add_start_button(self, grid: tuple[int, int]):
        self.env_elements.add_button(grid, 'Start AI', self.execute_start_command)

    def add_max_steps_spinbox(self, grid: tuple[int, int]):
        self.max_steps_spinbox_data = self.env_elements.add_spinbox(grid, 'Max steps', self.env_preferences.env_max_steps, increment=1000)

    def add_use_seed_checker(self, grid: tuple[int, int]):
        self.use_seed_checker_data = self.env_elements.add_checker(grid, 'Use seed', 'Seed')

    def add_seed_spinbox(self, grid: tuple[int, int]):
        self.seed_spinbox_data = self.env_elements.add_spinbox(grid, 'Seed', self.env_preferences.game_preferences.seed, increment=1000)
        
    def add_mines_num_spinbox(self, grid: tuple[int, int]):
        self.mines_num_spinbox_data = self.env_elements.add_spinbox(grid, 'Mines number', self.env_preferences.game_preferences.mines_num, increment=10)

    def add_field_size_spinbox(self, grid: tuple[int, int]):
        self.field_size_spinbox_data = self.env_elements.add_spinbox(grid, 'Field side size', self.env_preferences.game_preferences.field_size[0], increment=10)

    def execute_start_command(self):
        self.update_env()
        self.env_elements.tab_control.quit()
        self.env_elements.tab_control.destroy()
        self.start_command(self.env_preferences)

    def update_env(self):
        if not self.use_render_checker_data.variable.get():
            self.render_mode_box_data.element['state'] = 'disabled'
        else:
            self.render_mode_box_data.element['state'] = 'readonly'
        
        if not self.use_seed_checker_data.variable.get():
            self.seed_spinbox_data.element['state'] = 'disabled'
        else:
            self.seed_spinbox_data.element['state'] = 'normal'

    def update_env_preferences(self):
        self.env_preferences.game_preferences.mines_num = self.mines_num_spinbox_data.variable.get()
        field_size = (self.field_size_spinbox_data.variable.get(), self.field_size_spinbox_data.variable.get(),)
        self.env_preferences.game_preferences.field_size = field_size
        self.env_preferences.game_preferences.seed = self.seed_spinbox_data.variable.get() if self.use_seed_checker_data.variable.get() else None
        self.env_preferences.use_render = self.use_render_checker_data.variable.get()
        self.env_preferences.render_mode = self.render_mode_box_data.element.get()
    
    def add_env_to_window(self):
        elements = [
            self.add_field_size_spinbox,
            self.add_mines_num_spinbox,
            self.add_use_seed_checker,
            self.add_seed_spinbox,
            self.add_use_render_checker,
            self.add_render_mode_box,
            self.add_start_button
        ]
        self.env_elements.pack_tab(elements)

