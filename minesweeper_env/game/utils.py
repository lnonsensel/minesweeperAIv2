

class MinesweeperUtils:
    def __init__(self, cell_size, utility_area_size, field_area_size):
        self.cell_size = cell_size
        self.utility_area_size = utility_area_size
        self.field_area_size = field_area_size

    def window_to_cell_coords(self, window_coords: tuple[int, int]) -> tuple[int, int]:
        y = window_coords[0] // self.cell_size
        x = (window_coords[1] - self.utility_area_size[1])// self.cell_size
        return tuple(map(int, (x, y)))

    def cell_to_window_coords(self, cell_coords: tuple[int, int]) -> tuple[int, int]:
        y = cell_coords[0] * self.cell_size + self.utility_area_size[1]
        x = cell_coords[1] * self.cell_size 
        return tuple(map(int, (x, y)))
