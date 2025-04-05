from .state import State
from .game.direction import Direction
from random import choices
from typing import Dict

class Policy:
    def __init__(self):
        self.policy: Dict[State, Direction] = {}
        self.exploration = 0.3
        self.improvements = 0
        
    def get_action(self, state: State, current_direction: Direction):
        available = current_direction.get_available()
        if (not state in self.policy):
            return choices(available)[0]
        weights = [self.exploration / 2, self.exploration / 2, self.exploration / 2]
        try:
            weights[available.index(self.policy[state])] = 1 - self.exploration
            return choices(available, weights=weights)[0]
        except:
            return choices(available)[0]
        
    
    
    def improve(self, rewards):
        for state in rewards.keys():
            isUniformDistribution = True

            actions = list(rewards[state].keys())

            best_reward = rewards[state][actions[0]]["reward"]
            best_action = actions[0]

            for i in range(1, len(actions)):
                average_reward = rewards[state][actions[i]]["reward"]
                if (average_reward > best_reward):
                    best_reward = average_reward
                    best_action = actions[i]
                    isUniformDistribution = False
                elif (average_reward < best_reward):
                    isUniformDistribution = False
            if (not isUniformDistribution):
                self.policy[state] = best_action
        if (self.exploration > 0):
            self.exploration -= 0.001
        if (self.exploration < 0):
            self.exploration = 0
        print("Exploration rate: " + str(self.exploration))

    def __str__(self):
        s = "--------------------------\n"

        for state in self.policy.keys():
            s += str(state) + "\n"
            s += str(self.policy[state]) + "\n\n"

        s += "---------------------------\n"
            
        return s