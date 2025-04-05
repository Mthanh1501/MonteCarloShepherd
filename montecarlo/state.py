from .game.direction import ComplexDirection

class StateAction:
    def __init__(self, state, action):
        self.state = state
        self.action = action

    def __str__(self):
        s = ""
        s += str(self.state) + "\n"
        s += str(self.action) + "\n"
        return s

class State:
    def __init__(self, sheep_direction: ComplexDirection, facing_queue):
        self.sheep_direction = sheep_direction
        self.facing_queue = facing_queue

    def as_attack(self):
        return State(self.sheep_direction, set())
    
    def as_defense(self):
        return State(ComplexDirection.LEFT, self.facing_queue)
    
    def __eq__(self, other):
        return self.sheep_direction == other.sheep_direction and self.facing_queue == other.facing_queue
    
    def __hash__(self):
        return hash((self.sheep_direction, tuple(self.facing_queue)))
        
    def __str__(self):
        str_facing_queue = ""
        for direction in self.facing_queue:
            str_facing_queue += str(direction) + " "
        s = ""
        s += "Sheep on: " + str(self.sheep_direction) + "\n"
        s += "Facing queue: " + str_facing_queue + "\n"
        return s