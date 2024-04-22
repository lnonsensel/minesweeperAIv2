
# ENV PREFERENCES

field_size = (10, 10)
mines_num = 3
use_render = True
seed = 8128376198273
# render_mode = 'human'
render_mode = 'rgb_array'
max_steps = field_size[0] * field_size[1] // 2
env_kwargs = {'field_size': field_size,
              'mines_num': mines_num,
              'use_render': use_render,
              'seed': seed,
              'render_mode': render_mode,
              'max_steps': max_steps}


model_filename = 'dqn.pt'


# TRAINING PREFERENCES

max_steps = int(1e6)
eval_interval = 10000
warmup_steps = 1000
target_update_interval = 10000
