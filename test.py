import pytest
import numpy as np
import torch
from minesweeper_env.game.generator import MinesweeperGenerator
from minesweeper_env.game.scanner import MinesweeperScanner
from minesweeper_env.game.minesweeper_game import Minesweeper
from agent.dqn import DQN
from agent.replaybuffer import ReplayBuffer


class TestGenerator:
    def test_field_shape(self):
        gen = MinesweeperGenerator()
        field = gen.generate_field((9, 9), (0, 0), 10, seed=42)
        assert field.shape == (9, 9)

    def test_mine_count(self):
        gen = MinesweeperGenerator()
        field = gen.generate_field((9, 9), (0, 0), 10, seed=42)
        assert np.sum(field == 1.) == 10

    def test_start_cell_safe(self):
        gen = MinesweeperGenerator()
        field = gen.generate_field((9, 9), (4, 4), 10, seed=42)
        assert field[4, 4] != 1.

    def test_rectangular_field(self):
        gen = MinesweeperGenerator()
        field = gen.generate_field((5, 10), (0, 0), 5, seed=1)
        assert field.shape == (5, 10)
        assert np.sum(field == 1.) == 5


class TestScanner:
    def setup_method(self):
        gen = MinesweeperGenerator()
        self.field = gen.generate_field((9, 9), (0, 0), 10, seed=42)
        self.scanner = MinesweeperScanner(self.field)

    def test_get_cell_value_mine(self):
        mine_positions = np.argwhere(self.field == 1.)
        assert len(mine_positions) > 0
        y, x = mine_positions[0]
        assert self.scanner.get_cell_value((y, x)) == -3.

    def test_get_cell_value_safe_range(self):
        safe_positions = np.argwhere(self.field == 0.)
        for y, x in safe_positions[:5]:
            val = self.scanner.get_cell_value((y, x))
            assert 0 <= val <= 8, f"Unexpected value {val} at ({y},{x})"

    def test_get_neighbours_corner(self):
        neighbours = self.scanner.get_neighbours((0, 0), (9, 9))
        assert all(0 <= r < 9 and 0 <= c < 9 for r, c in neighbours)
        assert len(neighbours) <= 3

    def test_get_neighbours_center(self):
        neighbours = self.scanner.get_neighbours((4, 4), (9, 9))
        assert len(neighbours) == 8


class TestMinesweeper:
    def setup_method(self):
        self.game = Minesweeper((9, 9), 10, use_render=False, seed=42)

    def test_initial_state(self):
        assert self.game.game_lost is False
        assert self.game.game_won is False
        assert (self.game.opened_field == -2.).all()

    def test_first_click_generates_field(self):
        assert self.game.field is None
        self.game.play_action((1, 4, 4))
        assert self.game.field is not None

    def test_left_click_opens_cell(self):
        self.game.play_action((1, 4, 4))
        assert self.game.opened_field[4, 4] != -2.

    def test_first_click_always_safe(self):
        # Field is generated so start cell is never a mine
        self.game.play_action((1, 4, 4))
        assert self.game.game_lost is False

    def test_right_click_places_flag(self):
        self.game.play_action((1, 0, 0))  # generate field
        self.game.play_action((0, 8, 8))  # flag a closed cell
        assert self.game.opened_field[8, 8] == -1.

    def test_right_click_removes_flag(self):
        self.game.play_action((1, 0, 0))
        self.game.play_action((0, 8, 8))  # place flag
        self.game.play_action((0, 8, 8))  # remove flag
        assert self.game.opened_field[8, 8] == -2.

    def test_last_action_type_tracked(self):
        self.game.play_action((1, 4, 4))
        assert self.game.last_action_type == 1
        self.game.play_action((0, 8, 8))
        assert self.game.last_action_type == 0

    def test_reset_clears_state(self):
        self.game.play_action((1, 4, 4))
        self.game.reset_game()
        assert self.game.field is None
        assert (self.game.opened_field == -2.).all()
        assert self.game.last_action_type is None


class TestDQN:
    def setup_method(self):
        self.dqn = DQN(state_dim=(9, 9), action_dim=162)

    def test_get_channels_shape(self):
        field = torch.full((9, 9), -2.0)
        channels = self.dqn.get_channels(field)
        assert channels.shape == (3, 9, 9)

    def test_get_channels_all_closed(self):
        field = torch.full((9, 9), -2.0)
        channels = self.dqn.get_channels(field)
        assert (channels[1] == 0.0).all()   # nothing opened
        assert (channels[2] == 0.0).all()   # no flags

    def test_get_channels_flagged_cell(self):
        field = torch.full((9, 9), -2.0)
        field[3, 3] = -1.
        channels = self.dqn.get_channels(field)
        assert channels[2, 3, 3] == 1.0   # flag channel
        assert channels[1, 3, 3] == 1.0   # marked as "open" in channel 1

    def test_get_batch_channels_matches_single(self):
        field = torch.full((9, 9), -2.0)
        field[0, 0] = 3.0
        field[1, 1] = -1.0
        single = self.dqn.get_channels(field).unsqueeze(0)
        batch = self.dqn.get_batch_channels(field.to(self.dqn.device).unsqueeze(0))
        assert torch.allclose(single.to(self.dqn.device), batch)

    def test_get_batch_channels_shape(self):
        fields = torch.full((4, 9, 9), -2.0).to(self.dqn.device)
        result = self.dqn.get_batch_channels(fields)
        assert result.shape == (4, 3, 9, 9)

    def test_act_returns_valid_action(self):
        state = np.full((9, 9), -2.0, dtype=np.float32)
        action = self.dqn.act(state, training=False)
        assert isinstance(action, tuple)
        assert len(action) == 3
        action_type, y, x = action
        assert action_type in (0, 1)
        assert 0 <= y < 9
        assert 0 <= x < 9

    def test_act_random_during_warmup(self):
        # During warmup (total_steps < warmup_steps) action is random
        state = np.full((9, 9), -2.0, dtype=np.float32)
        self.dqn.total_steps = 0
        action = self.dqn.act(state, training=True)
        assert isinstance(action, (int, np.integer))


class TestReplayBuffer:
    def setup_method(self):
        self.buf = ReplayBuffer((9, 9), (3,), max_size=100)

    def test_update_increments_size(self):
        s = np.zeros((9, 9), dtype=np.float32)
        self.buf.update(s, np.array([1, 4, 4]), 1.0, s, False)
        assert self.buf.size == 1

    def test_sample_shapes(self):
        for _ in range(20):
            s = np.zeros((9, 9), dtype=np.float32)
            self.buf.update(s, np.array([1, 0, 0]), 1.0, s, False)
        states, actions, rewards, next_states, dones = self.buf.sample(4)
        assert states.shape == (4, 9, 9)
        assert actions.shape == (4, 3)
        assert rewards.shape == (4, 1)

    def test_circular_overflow(self):
        for i in range(150):
            s = np.full((9, 9), float(i), dtype=np.float32)
            self.buf.update(s, np.array([0, 0, 0]), float(i), s, False)
        assert self.buf.size == 100  # capped at max_size
        assert self.buf.ptr == 50    # wrapped around


class TestRewardConfig:
    def test_defaults_exist(self):
        from minesweeper_env.game.config import RewardConfig
        rc = RewardConfig()
        assert rc.loss_penalty < 0
        assert rc.win_base > 0
        assert rc.correct_flag_reward > 0
        assert rc.wrong_flag_penalty < 0

    def test_custom_values(self):
        from minesweeper_env.game.config import RewardConfig
        rc = RewardConfig(loss_penalty=-100.0, win_base=200.0)
        assert rc.loss_penalty == -100.0
        assert rc.win_base == 200.0
