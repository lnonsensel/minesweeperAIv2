from dataclasses import dataclass

@dataclass
class AgentPreferences:
    state_dim: tuple[int, int] | int | None = None
    action_dim: int | None = None
    lr: float = 0.0001
    epsilon: float = 1.0
    epsilon_min: float = 0.1
    gamma: float = 0.99
    batch_size: int = 32
    warmup_steps: int = 1000
    buffer_size: int = 50000
    tau: float = 0.005
    epsilon_decay_steps: int = 150000
