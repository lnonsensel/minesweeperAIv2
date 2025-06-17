from minesweeper_env.game.generator import MinesweeperGenerator
from minesweeper_env.game.scanner import MinesweeperScanner
from agent.cnn import MinesweeperAgent
import torch
import numpy as np
import random
    # gen = MinesweeperGenerator()
    # field = gen.generate_field((9, 9), (0,0), 50)

    # scanner = MinesweeperScanner(field)
    # scanner.get_cell_value((0,0))

def get_channels(field):
    field_scanned = np.zeros(field.shape, np.float16)
    opened_cells = np.ones(field.shape, dtype=np.float16)
    flag_cells = np.zeros(field.shape, dtype=np.float16)
    for y in range(field.shape[1]):
        for x in range(field.shape[0]):
            value = field[y][x]
            field_scanned[y,x] = -1. if value == -3. else value
            field_scanned[y,x] = 0. if value == -2. else value
            flag_cells[y,x] = 1. if value == -1. else 0.
            opened_cells[y,x] = 0. if value == -2. else 1.

    channel_0 = field_scanned
    channel_1 = opened_cells
    channel_2 = flag_cells
    return np.asarray([channel_0, channel_1, channel_2])

def concatenate_channels(channels):
    field_shape = channels[0].shape
    full_field = np.zeros((*field_shape, 3), dtype=np.float16)
    full_field[:,:,0] = channels[0] / 8.
    full_field[:,:,1] = channels[1]
    full_field[:,:,2] = channels[2]
    return full_field

def create_field_batch(batch_size = 1):
    batch = []
    for i in range(batch_size):
        full_field = np.array(get_channels())
        batch.append(full_field)
    return np.array(batch)

batch = create_field_batch(5)
tnsr = torch.Tensor(batch)

agnt = MinesweeperAgent(5)
responce = agnt(tnsr)

def get_max_index(tensor: torch.Tensor):
    argmax = torch.argmax(tensor).item()
    y, x = tensor.shape
    y_val = argmax // y
    x_val = argmax % x
    return (y_val, x_val)

def get_action_from_responce(responce):
    max_arg_click = get_max_index(responce[0])
    max_arg_flag = get_max_index(responce[1])

    if_click = torch.max(responce[0]).item() > torch.max(responce[1]).item()
    max_arg = max_arg_click if if_click else max_arg_flag
    action = (int(if_click), *max_arg)
    return action


if __name__ == '__main__':
    for batch_rsp in responce:
        print(get_action_from_responce(batch_rsp))