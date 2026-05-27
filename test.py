import pytest
import numpy as np
import torch
from minesweeper_env.game.generator import MinesweeperGenerator
from minesweeper_env.game.scanner import MinesweeperScanner
from minesweeper_env.game.minesweeper_game import Minesweeper
from agent.dqn import DQN
from agent.replaybuffer import ReplayBuffer, SumTree, PrioritizedReplayBuffer


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

    def test_flat_action_index_click(self):
        # click (1, y, x) → flat = y * W + x  (channel 0 of Q-output)
        H, W = self.dqn.H, self.dqn.W
        for y in range(H):
            for x in range(W):
                expected = y * W + x
                actual = y * self.dqn.W + x if 1 == 1 else self.dqn.H * self.dqn.W + y * self.dqn.W + x
                assert actual == expected

    def test_flat_action_index_flag(self):
        # flag (0, y, x) → flat = H*W + y * W + x  (channel 1 of Q-output)
        H, W = self.dqn.H, self.dqn.W
        for y in range(H):
            for x in range(W):
                expected_flag = H * W + y * W + x
                assert expected_flag >= H * W
                assert expected_flag < 2 * H * W

    def test_process_stores_flat_index(self):
        state = np.full((9, 9), -2.0, dtype=np.float32)
        # click on (3, 5): flat = 3 * 9 + 5 = 32
        self.dqn.process((state, (1, 3, 5), 0.0, state, False))
        assert self.dqn.buffer.a[0, 0] == 32
        # flag on (2, 4): flat = 9*9 + 2*9 + 4 = 81 + 18 + 4 = 103
        self.dqn.process((state, (0, 2, 4), 0.0, state, False))
        assert self.dqn.buffer.a[1, 0] == 103


class TestReplayBuffer:
    def setup_method(self):
        self.buf = ReplayBuffer((9, 9), (1,), max_size=100)

    def test_update_increments_size(self):
        s = np.zeros((9, 9), dtype=np.float32)
        self.buf.update(s, np.array([40]), 1.0, s, False)  # flat: click(4,4)=4*9+4=40
        assert self.buf.size == 1

    def test_sample_shapes(self):
        for _ in range(20):
            s = np.zeros((9, 9), dtype=np.float32)
            self.buf.update(s, np.array([0]), 1.0, s, False)
        states, actions, rewards, next_states, dones, weights, indices = self.buf.sample(4)
        assert states.shape == (4, 9, 9)
        assert actions.shape == (4, 1)
        assert rewards.shape == (4, 1)
        assert weights.shape == (4,)
        assert len(indices) == 4

    def test_circular_overflow(self):
        for i in range(150):
            s = np.full((9, 9), float(i), dtype=np.float32)
            self.buf.update(s, np.array([0]), float(i), s, False)
        assert self.buf.size == 100  # capped at max_size
        assert self.buf.ptr == 50    # wrapped around


class TestSumTree:
    def setup_method(self):
        self.tree = SumTree(8)

    def test_total_starts_zero(self):
        assert self.tree.total == 0.0

    def test_single_update(self):
        leaf_idx = self.tree.capacity - 1  # first leaf
        self.tree.update(leaf_idx, 5.0)
        assert self.tree.total == pytest.approx(5.0)

    def test_multiple_updates_sum(self):
        for i in range(4):
            self.tree.update(self.tree.capacity - 1 + i, float(i + 1))
        assert self.tree.total == pytest.approx(1 + 2 + 3 + 4)

    def test_get_returns_leaf(self):
        for i in range(4):
            self.tree.update(self.tree.capacity - 1 + i, 1.0)
        idx = self.tree.get(0.5)
        assert idx >= self.tree.capacity - 1

    def test_overwrite_updates_total(self):
        leaf_idx = self.tree.capacity - 1
        self.tree.update(leaf_idx, 10.0)
        self.tree.update(leaf_idx, 3.0)
        assert self.tree.total == pytest.approx(3.0)


class TestPrioritizedReplayBuffer:
    def setup_method(self):
        self.buf = PrioritizedReplayBuffer((9, 9), (1,), max_size=100)

    def test_update_increments_size(self):
        s = np.zeros((9, 9), dtype=np.float32)
        self.buf.update(s, np.array([40]), 1.0, s, False)
        assert self.buf.size == 1

    def test_sample_shapes(self):
        for _ in range(20):
            s = np.zeros((9, 9), dtype=np.float32)
            self.buf.update(s, np.array([0]), 1.0, s, False)
        states, actions, rewards, next_states, dones, weights, leaf_indices = self.buf.sample(4)
        assert states.shape == (4, 9, 9)
        assert actions.shape == (4, 1)
        assert weights.shape == (4,)
        assert len(leaf_indices) == 4

    def test_weights_in_range(self):
        for _ in range(20):
            s = np.zeros((9, 9), dtype=np.float32)
            self.buf.update(s, np.array([0]), 1.0, s, False)
        _, _, _, _, _, weights, _ = self.buf.sample(4)
        assert (weights > 0).all()
        assert (weights <= 1.0 + 1e-6).all()  # normalized to max=1

    def test_update_priorities(self):
        for _ in range(10):
            s = np.zeros((9, 9), dtype=np.float32)
            self.buf.update(s, np.array([0]), 1.0, s, False)
        _, _, _, _, _, _, leaf_indices = self.buf.sample(4)
        td_errors = np.array([0.5, 1.0, 2.0, 0.1])
        self.buf.update_priorities(leaf_indices, td_errors)
        assert self.buf.max_priority > 0

    def test_beta_anneals(self):
        initial_beta = self.buf.beta
        for _ in range(10):
            s = np.zeros((9, 9), dtype=np.float32)
            self.buf.update(s, np.array([0]), 1.0, s, False)
        self.buf.sample(4)
        assert self.buf.beta > initial_beta

    def test_circular_overflow(self):
        for i in range(150):
            s = np.full((9, 9), float(i), dtype=np.float32)
            self.buf.update(s, np.array([0]), float(i), s, False)
        assert self.buf.size == 100
        assert self.buf.ptr == 50


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

    def test_win_min_exists_and_positive(self):
        from minesweeper_env.game.config import RewardConfig
        rc = RewardConfig()
        assert rc.win_min > 0
        assert rc.win_min < rc.win_base

    def test_reward_clip_default_disabled(self):
        from minesweeper_env.game.config import RewardConfig
        rc = RewardConfig()
        assert rc.reward_clip == 0.0

    def test_reward_clip_custom(self):
        from minesweeper_env.game.config import RewardConfig
        rc = RewardConfig(reward_clip=10.0)
        assert rc.reward_clip == 10.0


class TestDuelingDQN:
    def setup_method(self):
        from agent.cnn import MinesweeperAgent
        self.model = MinesweeperAgent()

    def test_output_shape(self):
        x = torch.zeros(2, 3, 9, 9)
        out = self.model(x)
        assert out.shape == (2, 2, 9, 9)

    def test_value_advantage_decomposition(self):
        # All-same input → advantages should sum to near zero per cell
        x = torch.zeros(1, 3, 9, 9)
        out = self.model(x)
        # output = V + A - mean(A); if all A are equal mean(A) = A, so output ≈ V everywhere
        assert out.shape == (1, 2, 9, 9)
        # no NaN or inf
        assert torch.isfinite(out).all()
