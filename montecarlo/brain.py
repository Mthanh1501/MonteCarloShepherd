import random
import numpy as np
import shap
from .policy import Policy
from .state import State, StateAction
from .game.direction import ComplexDirection, Direction
from contextlib import redirect_stdout, redirect_stderr
import io

class Brain:
    def __init__(self, gamma):
        self.current_policy = Policy()
        self.values = {}
        self.rewards = {}
        self.history = []
        self.gamma = gamma
        self.reward_history = []
        self.state_history = []
        self.action_history = []

    def choose_direction(self, state, current_direction) -> Direction:
        direction = self.current_policy.get_action(state, current_direction)
        self.history.append([StateAction(state, direction), 0])
        self.state_history.append(self._state_to_vector(state))
        self.action_history.append(direction)
        return direction

    def _state_to_vector(self, state):
        """Chuyển trạng thái thành vector để SHAP xử lý, xử lý trường hợp None"""
        if state.sheep_direction is None:
            # Gán giá trị mặc định nếu sheep_direction là None
            sheep_dir_idx = list(ComplexDirection).index(ComplexDirection.RIGHT)  # Mặc định là RIGHT
        else:
            sheep_dir_idx = list(ComplexDirection).index(state.sheep_direction)
        
        facing_queue_str = "".join([str(list(Direction).index(d)) for d in state.facing_queue])
        facing_queue_val = int(facing_queue_str) if facing_queue_str else 0
        return np.array([sheep_dir_idx, facing_queue_val])

    def _predict_policy(self, states):
        """Hàm dự đoán hướng từ trạng thái cho SHAP"""
        actions = []
        for state_vec in states:
            sheep_dir = ComplexDirection(list(ComplexDirection)[int(state_vec[0])])
            facing_queue = set()
            if state_vec[1] > 0:
                fq_str = str(int(state_vec[1]))
                facing_queue = {Direction(int(d)) for d in fq_str}
            state = State(sheep_dir, facing_queue)
            action = self.current_policy.get_action(state, Direction.RIGHT)
            actions.append(list(Direction).index(action))
        return np.array(actions)

    def _interpret_shap_values(self, shap_values, state, action, current_direction):
        """Diễn giải giá trị SHAP thành câu trả lời tự nhiên"""
        sheep_dir_str = str(state.sheep_direction if state.sheep_direction else "Unknown").replace("ComplexDirection.", "")
        facing_queue_str = " ".join([str(d).replace("Direction.", "") for d in state.facing_queue]) or "None"
        action_str = str(action).replace("Direction.", "")
        current_dir_str = str(current_direction).replace("Direction.", "")

        shap_sheep = shap_values[0]
        shap_queue = shap_values[1]

        explanation = f"Người chăn cừu chọn hướng {action_str} vì:\n"
        if abs(shap_sheep) > 0.1:
            if shap_sheep > 0:
                explanation += f"- Cừu ở hướng {sheep_dir_str} khuyến khích di chuyển sang {action_str} (ảnh hưởng: {shap_sheep:+.2f}).\n"
            else:
                explanation += f"- Cừu ở hướng {sheep_dir_str} không khuyến khích {action_str}, nhưng các yếu tố khác mạnh hơn (ảnh hưởng: {shap_sheep:+.2f}).\n"
        else:
            explanation += f"- Hướng của cừu ({sheep_dir_str}) không ảnh hưởng nhiều đến quyết định này (ảnh hưởng: {shap_sheep:+.2f}).\n"

        if abs(shap_queue) > 0.1:
            if shap_queue > 0:
                explanation += f"- Các hướng bị chặn ({facing_queue_str}) hỗ trợ việc chọn {action_str} (ảnh hưởng: {shap_queue:+.2f}).\n"
            else:
                explanation += f"- Các hướng bị chặn ({facing_queue_str}) cản trở {action_str}, nhưng không đủ mạnh (ảnh hưởng: {shap_queue:+.2f}).\n"
        else:
            explanation += f"- Các hướng bị chặn ({facing_queue_str}) không ảnh hưởng đáng kể (ảnh hưởng: {shap_queue:+.2f}).\n"

        explanation += f"- Hướng hiện tại là {current_dir_str}, giúp việc chuyển sang {action_str} hợp lý hơn.\n"
        return explanation

    def explain_action(self, state, action, current_direction):
        """Giải thích hành động bằng SHAP, thêm Exploration rate và tắt tqdm"""
        action_str = str(action).replace("Direction.", "")
        sheep_direction_str = str(state.sheep_direction if state.sheep_direction else "Unknown").replace("ComplexDirection.", "")
        facing_queue_str = " ".join([str(d).replace("Direction.", "") for d in state.facing_queue])

        if not self.state_history:
            return f"Chọn hướng: {action_str}\nKhông đủ dữ liệu để giải thích SHAP."

        background_data = np.array(self.state_history[-100:])
        current_state_vec = self._state_to_vector(state).reshape(1, -1)

        explainer = shap.KernelExplainer(self._predict_policy, background_data)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            shap_values = explainer.shap_values(current_state_vec, nsamples=50)

        exploration_rate = f"Exploration rate: {self.current_policy.exploration:.3f}"

        explanation = f"Chọn hướng: {action_str}\n"
        explanation += f"Sheep Direction: {sheep_direction_str}\n"
        explanation += f"Facing Queue: {facing_queue_str if facing_queue_str else 'None'}\n"
        explanation += f"{exploration_rate}\n"
        explanation += "Giải thích SHAP:\n"
        explanation += f"- Sheep Direction: {shap_values[0][0]:+.2f}\n"
        explanation += f"- Facing Queue: {shap_values[0][1]:+.2f}\n"
        explanation += "\n" + self._interpret_shap_values(shap_values[0], state, action, current_direction)

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