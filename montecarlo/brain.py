import random
from .policy import Policy
from .state import StateAction
from .game.direction import ComplexDirection, Direction

class Brain:
    def __init__(self, gamma):
        self.current_policy = Policy()
        self.values = {}
        self.rewards = {}
        self.history = []
        self.gamma = gamma

    def choose_direction(self, state, current_direction) -> Direction:
        direction = self.current_policy.get_action(state, current_direction)
        self.history.append([StateAction(state, direction), 0])
        return direction

    def explain_action(self, state, action, current_direction):
        available = current_direction.get_available()
        # Loại bỏ tiền tố Direction. và ComplexDirection.
        action_str = str(action).replace("Direction.", "")
        sheep_direction_str = str(state.sheep_direction).replace("ComplexDirection.", "")
        
        base_info = "Chọn hướng: {}\nVị trí cừu: {}\nHướng bị chặn: {}\nMức độ Khám phá: {:.3f}".format(
            action_str,
            sheep_direction_str,
            state.facing_queue,
            self.current_policy.exploration
        )

        if state not in self.current_policy.policy:
            return base_info + "\nLựa chọn: Ngẫu nhiên vì chưa có chính sách\nLý do: Chưa có dữ liệu để đánh giá hướng đi"

        if random.random() < self.current_policy.exploration:
            return base_info + "\nLựa chọn: Khám phá\nLý do: Thử nghiệm hướng mới để tìm phần thưởng tốt hơn"

        reward_dict = self.rewards.get(state, {})
        reward_info = []
        for direction in available:
            reward = reward_dict.get(direction, {}).get("reward", 0)
            direction_str = str(direction).replace("Direction.", "")
            reward_info.append(f"{direction_str}: {reward:.2f}")
        best_reward = reward_dict.get(action, {}).get("reward", 0)
        explanation = (
            base_info +
            "\nPhần thưởng: {:.2f}".format(best_reward) +
            "\nPhần thưởng các hướng: " + ", ".join(reward_info) +
            "\nLý do: Hướng này có phần thưởng cao nhất"
        )
        return explanation

    def add_reward(self, reward):
        self.history[-1][1] = reward

    def evaluate(self):
        for i in range(0, len(self.history)):
            reward = 0
            counter = 0
            for j in range(i, len(self.history)):
                reward += self.history[j][1] * (self.gamma ** counter)
                counter += 1

            state_action = self.history[i][0]
            if state_action.state not in self.rewards:
                self.rewards[state_action.state] = {}
            if state_action.action not in self.rewards[state_action.state]:
                self.rewards[state_action.state][state_action.action] = {"reward": 0, "count": 0}

            old_reward = self.rewards[state_action.state][state_action.action]["reward"]
            count = self.rewards[state_action.state][state_action.action]["count"] + 1
            self.rewards[state_action.state][state_action.action]["count"] = count
            self.rewards[state_action.state][state_action.action]["reward"] = old_reward + (reward - old_reward) / count

        self.history = []
        self.current_policy.improve(self.rewards)