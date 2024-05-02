from menu.menu import MinesweeperAIMenu
from teacher.teacher import Teacher
from agent.dqn import DQN
from minesweeper_env.minenv import get_minenv
from dataclasses import asdict
from teacher.teacher import Teacher
from play_minesweeper import play_minesweeper
menu = MinesweeperAIMenu(env_entry_point=)
menu.run()


# kwargs = asdict(menu.preferences)
# env = get_minenv(**kwargs)
# dqn = DQN(env.observation_space.shape, env.action_space.n, batch_size=1, **kwargs)
# teacher = Teacher(dqn, env, kwargs)

# teacher.train()