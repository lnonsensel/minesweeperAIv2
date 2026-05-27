from menu.elements import TkinterElements
from agent.preferences import AgentPreferences

class AgentMenu:
    def __init__(self, tab_control) -> None:
        self.agent_preferences = AgentPreferences()
        self.agent_elements = TkinterElements(tab_control, 'Agent')

    def add_lr_spinbox(self, grid: tuple[int, int]):
        self.lr_spinbox_data = self.agent_elements.add_spinbox(grid, 'Learning rate', self.agent_preferences.lr, increment=0.0001)

    def add_epsilon_spinbox(self, grid: tuple[int, int]):
        self.epsilon_spinbox_data = self.agent_elements.add_spinbox(grid, 'Epsilon', self.agent_preferences.epsilon, increment=0.01)

    def add_epsilon_min_spinbox(self, grid: tuple[int, int]):
        self.epsilon_min_spinbox_data = self.agent_elements.add_spinbox(grid, 'Epsilon min', self.agent_preferences.epsilon_min, increment=0.01)

    def add_gamma_spinbox(self, grid: tuple[int, int]):
        self.gamma_spinbox_data = self.agent_elements.add_spinbox(grid, 'Gamma', self.agent_preferences.gamma, increment=0.01)

    def add_batch_size_spinbox(self, grid: tuple[int, int]):
        self.batch_size_spinbox_data = self.agent_elements.add_spinbox(grid, 'Batch size', self.agent_preferences.batch_size, increment=2)

    def add_warmup_steps_spinbox(self, grid: tuple[int, int]):
        self.warmup_steps_spinbox_data = self.agent_elements.add_spinbox(grid, 'Warmup steps', self.agent_preferences.warmup_steps, increment=1000)

    def add_buffer_size_spinbox(self, grid: tuple[int, int]):
        self.buffer_size_spinbox_data = self.agent_elements.add_spinbox(grid, 'Buffer size', self.agent_preferences.buffer_size, increment=1000)

    def add_tau_spinbox(self, grid: tuple[int, int]):
        self.tau_spinbox_data = self.agent_elements.add_spinbox(grid, 'Tau (soft update)', self.agent_preferences.tau, increment=0.001)

    def add_epsilon_decay_steps_spinbox(self, grid: tuple[int, int]):
        self.epsilon_decay_steps_spinbox_data = self.agent_elements.add_spinbox(grid, 'Epsilon decay steps', self.agent_preferences.epsilon_decay_steps, increment=10000)

    def update_preferences(self):
        self.agent_preferences.lr = self.lr_spinbox_data.variable.get()
        self.agent_preferences.epsilon = self.epsilon_spinbox_data.variable.get()
        self.agent_preferences.epsilon_min = self.epsilon_min_spinbox_data.variable.get()
        self.agent_preferences.gamma = self.gamma_spinbox_data.variable.get()
        self.agent_preferences.batch_size = self.batch_size_spinbox_data.variable.get()
        self.agent_preferences.warmup_steps = self.warmup_steps_spinbox_data.variable.get()
        self.agent_preferences.buffer_size = self.buffer_size_spinbox_data.variable.get()
        self.agent_preferences.tau = self.tau_spinbox_data.variable.get()
        self.agent_preferences.epsilon_decay_steps = self.epsilon_decay_steps_spinbox_data.variable.get()

    def add_agent_to_menu(self):
        elements = [self.add_lr_spinbox,
                    self.add_epsilon_spinbox,
                    self.add_epsilon_min_spinbox,
                    self.add_gamma_spinbox,
                    self.add_batch_size_spinbox,
                    self.add_warmup_steps_spinbox,
                    self.add_buffer_size_spinbox,
                    self.add_tau_spinbox,
                    self.add_epsilon_decay_steps_spinbox]
        self.agent_elements.pack_tab(elements)
