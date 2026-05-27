# MinesweeperAIv2

Deep Q-Network (DQN) agent that learns to play Minesweeper.

![Menu](images/menu.png)
![Working Example](images/working.gif)

## Overview

The agent uses a convolutional neural network to process the game board and outputs Q-values for every possible action (open cell or place flag) at every position. Training is driven by a custom Gymnasium environment with a dense reward function.

**Architecture:**
- CNN with sliding window feature extraction (5×5 kernel, unfold-based)
- DQN with experience replay and a target network
- Two output channels: click map and flag map

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

## Usage

### Launch GUI (train / evaluate / play)

```bash
python main.py
# or
make train
```

The Tkinter menu lets you configure field size, mine count, agent hyperparameters, and training duration before starting.

### Play manually

```bash
python play_minesweeper.py
# or
make play
```

### Run tests

```bash
pytest test.py -v
# or
make test
```

### Enable debug logging

```bash
MINESWEEPER_DEBUG=1 python main.py
# or
make debug
```

## Project structure

```
agent/          DQN agent, CNN, replay buffer, hyperparameters
menu/           Tkinter configuration GUI
minesweeper_env/  Gymnasium environment + game logic
teacher/        Training loop, evaluator, checkpointing
evaluations/    Saved model checkpoints and training plots
```

## Checkpoints

Models are saved to `evaluations/` when a new best evaluation score is achieved. Each checkpoint is a full dict:

```python
{
  'network':       state_dict,
  'optimizer':     state_dict,
  'total_steps':   int,
  'epsilon':       float,
  'top_eval_score': float,
}
```

The filename encodes `{timestamp}_{steps}_{eval_score}_dqn.pt`.

## Reward shaping

All reward constants are in `minesweeper_env/game/config.py` under `RewardConfig` and can be tuned without touching game logic.
