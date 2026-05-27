from dataclasses import dataclass

# ENV PREFERENCES

field_size = (10, 10)
mines_num = 15
use_render = True
seed = 618297364
render_mode = 'human'
render_modes = ('info', 'human', 'none')
max_steps = field_size[0] * field_size[1] // 2
env_kwargs = {'field_size': field_size,
              'mines_num': mines_num,
              'use_render': use_render,
              'seed': seed,
              'render_mode': render_mode,
              'max_steps': max_steps}


model_filename = 'dqn.pt'
env_max_steps = 200


@dataclass
class RewardConfig:
    loss_penalty: float = -50.0
    win_base: float = 100.0
    win_speed_factor: float = 0.5
    step_penalty: float = -0.3
    repeat_click_penalty: float = -2.0
    safe_cell_reward: float = 1.5
    neighbor_info_factor: float = 0.8
    open_neighbor_reward: float = 0.3
    surround_bonus: float = 2.0
    surround_threshold: int = 4
    correct_flag_reward: float = 5.0
    mine_cluster_factor: float = 0.7
    wrong_flag_penalty: float = -4.0
    new_cell_factor: float = 0.4
    correct_flag_factor: float = 0.2
