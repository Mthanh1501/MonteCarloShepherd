from random import randint
from os import urandom
from .direction import Direction
from .grid import Grid
from .direction import ComplexDirection


class Sheep:
    def __init__(self, x_cell, y_cell):
        self.x_cell = x_cell
        self.y_cell = y_cell

class Shepperd:
    speed = 1

    def __init__(self, x_cell, y_cell, grid: Grid):
        self.x_cell = x_cell
        self.y_cell = y_cell
        self.grid = grid
        self.sheeps = 0

  

    def move(self, direction: Direction):
        if (direction == Direction.DOWN):
            self.y_cell = (self.y_cell + self.speed) % (self.grid.grid_side)
        elif (direction == Direction.UP):
            self.y_cell = (self.y_cell - self.speed) % (self.grid.grid_side)
        elif (direction == Direction.LEFT):
            self.x_cell = (self.x_cell - self.speed) % (self.grid.grid_side)
        elif (direction == Direction.RIGHT):
            self.x_cell = (self.x_cell + self.speed) % (self.grid.grid_side)

    

    def get_sheep_direction(self, sheep: Sheep):
        if (self.x_cell < sheep.x_cell):
            if (self.y_cell < sheep.y_cell):
                return ComplexDirection.BOTTOM_RIGHT
            elif (self.y_cell > sheep.y_cell):
                return ComplexDirection.TOP_RIGHT
            else:
                return ComplexDirection.RIGHT
        elif (self.x_cell > sheep.x_cell):
            if (self.y_cell < sheep.y_cell):
                return ComplexDirection.BOTTOM_LEFT
            elif (self.y_cell > sheep.y_cell):
                return ComplexDirection.TOP_LEFT
            else:
                return ComplexDirection.LEFT
        else:
            if (self.y_cell < sheep.y_cell):
                return ComplexDirection.DOWN
            elif (self.y_cell > sheep.y_cell):
                return ComplexDirection.UP


    def get_queue_directions(self, positions, current_direction: Direction):
        
        directions = set()

        #print(positions)

        for pos in positions:
            if (self.x_cell == pos[0]):
                if (self.y_cell - pos[1] == 1 and current_direction != Direction.DOWN):
                    directions.add(Direction.UP)
                elif (self.y_cell - pos[1] == -1 and current_direction != Direction.UP):
                    directions.add(Direction.DOWN)
            elif (self.y_cell == pos[1]):
                if (self.x_cell - pos[0] == 1 and current_direction != Direction.RIGHT):
                    directions.add(Direction.LEFT)
                elif (self.x_cell - pos[0] == -1 and current_direction != Direction.LEFT):
                    directions.add(Direction.RIGHT)

        
        #directions.remove(current_direction.get_opposite())

        return directions