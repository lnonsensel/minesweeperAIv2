from collections import deque
import numpy as np

class MinesweeperScanner:
    def __init__(self, field):
        self.field = field

    def get_neighbours(self, coords: tuple[int, int], field_shape: tuple[int, int]) -> tuple[tuple[int, int]]:
        neighbours = []
        for k in (-1, 0, 1):
            for i in (-1, 0, 1):
                neighbour = (coords[0] - i, coords[1]- k)
                if all(j >= 0 for j in neighbour)\
                    and neighbour != coords\
                    and (neighbour[1] < field_shape[0])\
                    and (neighbour[0] < field_shape[1]):

                    neighbours.append(neighbour)
        return neighbours

    def get_neighbours_with_zero(self, coords: tuple[int, int])-> list[tuple[int, int]]:
        queue = deque()
        queue.append(coords)
        zero_open = set()
        visited = set()
        while queue:
            cur_coords = queue.popleft()
            neis = self.get_neighbours(cur_coords, self.field.shape)
            zero_open.update(neis)
            visited.add(cur_coords)
            for nei in neis:
                if self.get_cell_value(nei) == 0 and nei not in queue and nei not in visited:
                    queue.append(nei)
        return list(zero_open)

    def get_cell_value(self, cell_coords: tuple[int, int]) -> float:
        '''
        -3 value returned for mine field
        '''
        if self.field[cell_coords[0]][cell_coords[1]] == 1.:
            return -3.
        cell_value = sum(self.field[neig] for neig in self.get_neighbours(cell_coords, self.field.shape))
        return cell_value