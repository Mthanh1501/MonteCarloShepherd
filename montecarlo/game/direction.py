from enum import Enum

class Direction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    def get_available(self):
        if (self.value == self.LEFT.value):
            return [self.LEFT, self.UP, self.DOWN]
        elif (self.value == self.RIGHT.value):
            return [self.RIGHT, self.UP, self.DOWN]
        elif (self.value == self.UP.value):
            return [self.LEFT, self.RIGHT, self.UP]
        elif (self.value == self.DOWN.value):
            return [self.LEFT, self.RIGHT, self.DOWN]
        
    def get_opposite(self):
        if (self.value == self.LEFT.value):
            return Direction.RIGHT
        elif (self.value == self.RIGHT.value):
            return Direction.LEFT
        elif (self.value == self.UP.value):
            return Direction.DOWN
        elif (self.value == self.DOWN.value):
            return Direction.UP
        

class ComplexDirection(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    TOP_RIGHT = 4
    TOP_LEFT = 5
    BOTTOM_RIGHT = 6
    BOTTOM_LEFT = 7