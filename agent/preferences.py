from dataclasses import dataclass

@dataclass
class AgentPreferences:
    state_dim: tuple[int, int] | int | None = None
    action_dim: int | None = None
    lr: int=0.00025
    epsilon: int=1.0
    epsilon_min: int=0.1
    gamma: int=0.99
    batch_size: int=32
    warmup_steps: int=5000
    buffer_size: int=int(1e5)
    target_update_interval: int=10000