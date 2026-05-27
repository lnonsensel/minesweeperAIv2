from dataclasses import dataclass
from minesweeper_env.preferences import MinesweeperEnvPreferences
from agent.preferences import AgentPreferences

@dataclass
class EvaluatorPreferences:
    model_filename: str = 'none'

@dataclass
class TeacherPreferences:
    env_preferences: MinesweeperEnvPreferences | None = None
    agent_preferences: AgentPreferences | None = None
    eval_interval: int = 10000
    learning_max_steps: int = 200000
    model_filename: str = 'dqn.pt'
    resume_from: str | None = None
    use_tensorboard: bool = False
