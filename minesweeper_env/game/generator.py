import numpy as np
import random

class MinesweeperGenerator:

    def generate_field(self,
                       shape: tuple[int, int],
                        start_cell: tuple[int, int],
                        mines_num: int,
                        seed: int | None = None) -> np.ndarray:

        field = np.zeros(shape, dtype='float32')
        used_cords = set()

        while len(used_cords) != mines_num:
            if seed is not None:
                random.seed(seed)
            coord_y = random.randint(0, shape[0] - 1)
            if seed is not None:
                random.seed(seed * 2)
            coord_x = random.randint(0, shape[1] - 1)
            new_coord = (coord_y, coord_x)
            if new_coord != start_cell:
                used_cords.add(new_coord)
            if seed is not None:
                seed += 1
        for coord in used_cords:
            field[coord[0]][coord[1]] = 1.

        return field