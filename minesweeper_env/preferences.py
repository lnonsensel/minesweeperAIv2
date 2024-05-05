from dataclasses import dataclass
from minesweeper_env.game.config import field_size, mines_num, use_render, seed, render_mode, env_max_steps, render_modes

@dataclass
class MinesweeperGamePreferences:
    field_size: tuple[int, int] = field_size 
    mines_num: int = mines_num
    seed: int = seed

@dataclass
class MinesweeperEnvPreferences:
    game_preferences: MinesweeperGamePreferences = MinesweeperGamePreferences()
    use_render: bool = use_render
    render_modes: list[str] = render_modes
    render_mode: str = render_mode
    env_max_steps: int = env_max_steps