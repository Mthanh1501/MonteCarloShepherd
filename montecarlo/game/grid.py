import math
from random import randint

class Grid:
    def __init__(self, grid_side, pixel_side):
        self.grid_side = grid_side
        self.pixel_side = pixel_side
        self.ratio = pixel_side / grid_side

    def to_cell(self, n):
        return math.floor((n / self.pixel_side) * self.grid_side)
    
    def from_cell(self, n):
        return n * self.ratio
    
    def random_cell(self):
        return randint(0, self.grid_side - 1)