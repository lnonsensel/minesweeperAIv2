# Architecture

A Double DQN agent that learns to play Minesweeper through dense reward shaping.
The environment is a custom [Gymnasium](https://gymnasium.farama.org/) wrapper around a pure-Python
Minesweeper game with optional Pygame rendering. The neural network is a fully-convolutional
PyTorch model trained with experience replay and soft target-network updates.

---

## Repository Layout

```
minesweeperAIv2/
│
├── minesweeper_env/          # Gymnasium environment
│   ├── game/
│   │   ├── generator.py      # Random mine-field generation
│   │   ├── scanner.py        # Neighbour queries, flood-fill, cell values
│   │   ├── minesweeper_game.py  # Core game logic (no ML)
│   │   ├── renderer.py       # Pygame board rendering
│   │   ├── config.py         # RewardConfig dataclass + global defaults
│   │   └── utils.py
│   ├── minenv.py             # MinesweeperEnv — gym.Env wrapper
│   └── preferences.py        # MinesweeperGamePreferences, MinesweeperEnvPreferences
│
├── agent/
│   ├── cnn.py                # MinesweeperAgent — fully-convolutional Q-network
│   ├── dqn.py                # DQN — Double DQN algorithm, ε-greedy, soft updates
│   ├── replaybuffer.py       # ReplayBuffer — circular experience buffer
│   └── preferences.py        # AgentPreferences dataclass
│
├── teacher/
│   ├── teacher.py            # Teacher — training loop, evaluation, checkpointing
│   ├── evaluator.py          # Evaluator — load checkpoint and demo-play
│   ├── preferences.py        # TeacherPreferences, EvaluatorPreferences
│   └── config.py             # MODELS_CHECKPOINTS_PATH
│
├── menu/                     # Tkinter GUI (configure and launch modes)
├── main.py                   # Entry point: GUI → train / evaluate / play
├── play_minesweeper.py       # Standalone manual Minesweeper (Pygame)
├── utils.py                  # @log decorator (optional timing output)
├── test.py                   # pytest suite (32 tests)
└── Makefile                  # train / play / test / debug targets
```

---

## Game Layer

### Field generation — `generator.py`

`MinesweeperGenerator.generate_field(shape, start_cell, mines_num, seed)` returns a
`float32` NumPy array where **mines = `1.`** and **safe cells = `0.`**.

Field generation is **deferred until the first click** — `Minesweeper._left_click_action`
calls the generator only when `self.field is None`, passing the clicked cell as `start_cell`.
This guarantees the first click is always safe.

### Queries — `scanner.py`

`MinesweeperScanner` operates on the raw `field` array:

| Method | Description |
|--------|-------------|
| `get_neighbours(coords, shape)` | Moore neighbourhood (≤8 cells), bounds-checked |
| `get_cell_value(coords)` | Returns `-3.` for a mine, `0–8` for safe (adjacent mine count) |
| `get_neighbours_with_zero(coords)` | BFS flood-fill: returns all cells reachable through zero-valued cells |

### Core game — `minesweeper_game.py`

`Minesweeper` is the base class inherited by `MinesweeperEnv`. Key responsibilities:

| Method | Description |
|--------|-------------|
| `reset_state()` | Sets `opened_field` to `-2.` (all closed), clears flags and last-action state |
| `play_action((type, y, x))` | Dispatches: `type=1` → left-click, `type=0` → flag toggle |
| `_left_click_action(coords)` | Opens cell; if value=0 applies BFS flood to open the region |
| `_right_click_action(coords)` | Toggles `opened_field[coords]` between `-2.` and `-1.`; updates `placed_good_flags` |
| `check_game_end()` | Sets `game_lost` / `game_won` |

---

## Board State Encoding

Two NumPy arrays are always maintained in parallel:

| Array | Closed | Flagged | Mine hit | Opened cell |
|-------|--------|---------|----------|-------------|
| `opened_field` — what the **agent sees** | `-2.` | `-1.` | `-3.` | `0–8` (neighbor mine count) |
| `field` — raw layout, **hidden** | `0.` | `0.` | `1.` | — |

`player_field` starts as all `1.` and mirrors `field`; a cell is set to `0.` when opened.
Win condition: `(player_field == field).all()` — every mine cell is still `1.` (unflagged or
flagged) and every safe cell has been opened to `0.`.

---

## Action Space

```
action_space = Discrete(2 × H × W)
```

Actions are encoded as a tuple `(action_type, y, x)`:

| `action_type` | Meaning |
|---------------|---------|
| `1` | Left-click (open) cell `(y, x)` |
| `0` | Toggle flag at cell `(y, x)` |

The first `H×W` indices correspond to left-clicks; the next `H×W` to flag toggles.
`MinesweeperEnv.get_actions()` builds the lookup table `actions[flat_index] = (type, y, x)`.

**Flat index used during training** (position in the Q-vector after `reshape(B, -1)`):

```
click (type=1):  flat = y × W + x
flag  (type=0):  flat = H×W + y×W + x
```

`DQN.process()` performs this conversion before writing to the replay buffer so that
`gather(1, a)` in `learn()` addresses the correct Q-value.

---

## Gymnasium Wrapper — `minenv.py`

`MinesweeperEnv` inherits both `Minesweeper` and `gym.Env`:

```
observation_space = Box(-4, 8, shape=(1, H, W))   # opened_field
action_space      = Discrete(2 × H × W)
```

`step(action)`:
1. Saves `previous_opened_field` for reward delta calculation.
2. Calls `play_action(action)`.
3. Calls `get_reward()` — updates `self.reward` (cumulative).
4. Returns `(obs, step_reward, terminated, truncated, {'score': self.reward})`
   where `step_reward = self.reward - self.last_reward`.

`reset()` resets cumulative `reward`, `step_ind`, and calls `reset_game()`.

---

## Reward Function

All constants are in `RewardConfig` (`minesweeper_env/game/config.py`).
`self.reward` is **cumulative**; each `step()` call returns only the delta.

| Event | Default value |
|-------|--------------|
| Game lost | `loss_penalty = -50.0` |
| Game won | `max(win_min=20.0, win_base=100.0 − step_ind × 0.5)` |
| Each step | `step_penalty = -0.3` |
| Repeat click (no state change) | `repeat_click_penalty = -2.0` |
| Open a new safe cell | `safe_cell_reward = +1.5` |
| Each already-open neighbour of opened cell | `open_neighbor_reward = +0.3` |
| Opened cell value × factor | `neighbor_info_factor = 0.8` per adjacent mine |
| 4+ neighbours open after this click | `surround_bonus = +2.0` (`surround_threshold = 4`) |
| Correct flag placed on a mine | `correct_flag_reward = +5.0` |
| Per mine adjacent to correctly flagged cell | `mine_cluster_factor = +0.7` |
| Wrong flag placed on a safe cell | `wrong_flag_penalty = -4.0` |
| Per newly revealed cell (bulk opens) | `new_cell_factor = +0.4` |
| Per net-new correctly placed flag (delta) | `correct_flag_factor = +0.2` |

The `correct_flag_factor` bonus is **delta-based** (uses `previous_good_flags` set from the
previous step) to prevent it from accumulating across every subsequent step.

---

## Neural Network — `agent/cnn.py`

### Input encoding

`DQN.get_channels(opened_field)` converts the single `(H, W)` board into `(3, H, W)`:

| Channel | Derivation | Represents |
|---------|------------|------------|
| 0 | `-3 → -1`, `-2 → 0`, otherwise as-is | Visible cell values |
| 1 | `-2 → 0`, else `1` | Open / closed mask |
| 2 | `-1 → 1`, else `0` | Flag mask |

`get_batch_channels(fields)` performs the same transformation vectorised over a batch.

### MinesweeperAgent

```
Input  (B, 3, H, W)
  Conv2d(3  → 32,  3×3, pad=1) → BatchNorm2d → ReLU
  Conv2d(32 → 64,  3×3, pad=1) → BatchNorm2d → ReLU
  Conv2d(64 → 64,  3×3, pad=1) → BatchNorm2d → ReLU
  Conv2d(64 → 2,   1×1)
Output (B, 2, H, W)
```

All 3×3 layers use `padding=1`, so spatial dimensions are preserved throughout.
Effective receptive field: **7×7**.
Output channel 0 = click Q-values; channel 1 = flag Q-values.

---

## DQN Agent — `agent/dqn.py`

### Algorithm

**Double DQN** with soft target-network updates (no hard copy):

```
TD target = r + γ · Q_target(s', argmax_a Q_online(s', ·))
```

Online network selects the action; target network evaluates its value.
This decoupling reduces Q-value overestimation.

Soft update after every learning step:

```
θ_target ← τ · θ_online + (1 − τ) · θ_target    (τ = 0.005)
```

### Key methods

| Method | Description |
|--------|-------------|
| `act(state, training)` | ε-greedy during training; greedy (tuple) at inference |
| `process(transition)` | Converts action tuple → flat index → buffer; calls `learn()` after warmup |
| `learn()` | Samples batch, computes Double-DQN target, MSE loss, grad-clip (norm=10), optimizer step |
| `get_channels(field)` | Single-board channel encoding |
| `get_batch_channels(fields)` | Vectorised batch channel encoding |
| `get_action_from_response(q)` | Picks `(type, y, x)` from `(1, 2, H, W)` Q output |

### Exploration schedule

ε decays **linearly** from `1.0` to `epsilon_min=0.1` over `epsilon_decay_steps=150000` steps.
At the default budget of 200k steps, the agent is fully exploiting for the final 50k steps.

During the first `warmup_steps=1000` steps the buffer is filled with random actions before
any gradient updates begin.

---

## Replay Buffer — `agent/replaybuffer.py`

Circular NumPy buffer of capacity `buffer_size=50000`:

| Array | Shape | dtype |
|-------|-------|-------|
| `s` | `(max_size, H, W)` | float32 |
| `a` | `(max_size, 1)` | int64 (flat index) |
| `r` | `(max_size, 1)` | float32 |
| `s_prime` | `(max_size, H, W)` | float32 |
| `terminated` | `(max_size, 1)` | float32 |

`sample(batch_size)` returns a tuple of five `torch.FloatTensor` tensors sampled uniformly.

---

## Training Pipeline — `teacher/teacher.py`

```
Teacher.train()
│
├── env.reset() → s
│
└── loop while total_steps ≤ learning_max_steps:
    ├── agent.act(s)
    │     warmup / ε-greedy → int (flat index)
    │     greedy            → (type, y, x) tuple
    ├── if int: lookup env.unwrapped.actions[a] → tuple
    ├── env.step(a) → s', r, terminated, truncated, info
    ├── agent.process((s, a, r, s', terminated))
    │     ├── flat_a = encode action
    │     ├── buffer.update(s, flat_a, r, s', terminated)
    │     ├── [after warmup] learn() + soft target update
    │     └── ε -= ε_decay
    ├── s = s'
    ├── if done: env.reset() → s
    │
    └── every eval_interval steps:
          ├── evaluate() → avg_return, win_rate
          ├── if best avg_return: checkpoint_model()
          └── every 10th eval: create_history_plot() + save history.csv
```

### Checkpoints

Files saved to `evaluations/` with name `{timestamp}_{steps}_{score}_{model_filename}`.

Checkpoint dict:

```python
{
    'network':       network.state_dict(),
    'optimizer':     optimizer.state_dict(),
    'total_steps':   int,
    'epsilon':       float,
    'top_eval_score': float,
}
```

`Evaluator.get_loaded_agent()` handles both this dict format and legacy bare state-dicts.

### Evaluation

`Teacher.evaluate(n_evals=5)` runs `n` full episodes with `training=False` (greedy, ε ignored).
Returns `avg_return` (mean cumulative reward). Also records `last_win_rate`.
A new checkpoint is saved only when `avg_return` strictly exceeds the previous best.

---

## Configuration Reference

| File | Dataclass | Notable defaults |
|------|-----------|-----------------|
| `minesweeper_env/preferences.py` | `MinesweeperGamePreferences` | `field_size=(10,10)`, `mines_num=15` |
| `minesweeper_env/preferences.py` | `MinesweeperEnvPreferences` | `render_mode='human'`, `env_max_steps=200` |
| `agent/preferences.py` | `AgentPreferences` | `lr=1e-4`, `gamma=0.99`, `batch_size=32`, `tau=0.005`, `epsilon_decay_steps=150000`, `warmup_steps=1000`, `buffer_size=50000` |
| `teacher/preferences.py` | `TeacherPreferences` | `eval_interval`, `learning_max_steps`, `model_filename` |
| `minesweeper_env/game/config.py` | `RewardConfig` | all 13 reward constants (see Reward Function section) |

---

## Entry Points

| Command | Description |
|---------|-------------|
| `make train` / `python main.py` | Tkinter GUI to configure and launch training |
| `make play` / `python play_minesweeper.py` | Manual Minesweeper with Pygame mouse controls |
| `make test` | `pytest test.py -v` — 32 unit tests |
| `make debug` | `MINESWEEPER_DEBUG=1 python main.py` — prints per-function timing via `@log` |
