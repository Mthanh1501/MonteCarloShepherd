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

    # def _interpret_shap_values(self, shap_values, state, action, current_direction):
    #     """Giải thích chi tiết tại sao người chăn cừu chọn hướng này, kể cả khi đi ngẫu nhiên."""
    #     # Chuẩn bị thông tin
    #     sheep_dir_str = str(state.sheep_direction if state.sheep_direction else "Không rõ").replace("ComplexDirection.", "")
    #     facing_queue_str = " ".join([str(d).replace("Direction.", "") for d in state.facing_queue]) or "Không có"
    #     action_str = str(action).replace("Direction.", "")

    #     # Giá trị SHAP
    #     shap_sheep = shap_values[0]  # Ảnh hưởng từ hướng cừu
    #     shap_queue = shap_values[1]  # Ảnh hưởng từ chướng ngại

    #     # Kiểm tra tỷ lệ đi ngẫu nhiên từ Policy
    #     exploration_rate = self.current_policy.exploration  # Giả sử Policy có thuộc tính này
    #     is_random = exploration_rate > 0.5  # Ngẫu nhiên nếu tỷ lệ khám phá cao (có thể điều chỉnh ngưỡng)

    #     # Bắt đầu giải thích
    #     explanation = f"Người chăn cừu chọn hướng {action_str}:\n"

    #     # Thêm thông tin ngẫu nhiên nếu có
    #     if is_random:
    #         explanation += f"- Lần này hướng {action_str} được chọn ngẫu nhiên (tỷ lệ khám phá: {exploration_rate:.2f}).\n"
    #     else:
    #         explanation += f"- Hướng {action_str} được chọn dựa trên đánh giá tình huống.\n"

    #     # Giải thích chi tiết từ hướng cừu (luôn thực hiện)
    #     if shap_sheep > 0:
    #         explanation += f"- Cừu ở hướng {sheep_dir_str} khiến {action_str} là lựa chọn tốt.\n"
    #         # Kiểm tra nếu ăn cừu hoặc gần cừu
    #         if action_str == sheep_dir_str:  # Giả sử action trùng hướng cừu là "ăn cừu"
    #             explanation += f"- Người chăn cừu chọn ăn cừu.\n"
    #         else:
    #             explanation += f"- Di chuyển tới {action_str} vì gần cừu hơn.\n"
    #     elif shap_sheep < 0:
    #         explanation += f"- Mặc dù cừu ở {sheep_dir_str} không ủng hộ, các yếu tố khác mạnh hơn.\n"
    #     else:
    #         explanation += f"- Hướng cừu ({sheep_dir_str}) không ảnh hưởng nhiều.\n"

    #     # Giải thích chi tiết từ chướng ngại (luôn thực hiện)
    #     if shap_queue > 0:
    #         explanation += f"- Các hướng bị chặn ({facing_queue_str}) đẩy người chăn cừu sang {action_str}.\n"
    #     elif shap_queue < 0:
    #         explanation += f"- Dù chướng ngại ({facing_queue_str}) cản trở, {action_str} vẫn được chọn.\n"
    #     else:
    #         explanation += f"- Chướng ngại ({facing_queue_str}) không tác động đáng kể.\n"

    #     return explanation

    # def explain_action(self, state, action, current_direction):
    #     """Giải thích hành động bằng SHAP, thêm Exploration rate và tắt tqdm"""
    #     action_str = str(action).replace("Direction.", "")
    #     sheep_direction_str = str(state.sheep_direction if state.sheep_direction else "Unknown").replace("ComplexDirection.", "")
    #     facing_queue_str = " ".join([str(d).replace("Direction.", "") for d in state.facing_queue])

    #     if not self.state_history:
    #         return f"Chọn hướng: {action_str}\nKhông đủ dữ liệu để giải thích SHAP."

    #     background_data = np.array(self.state_history[-100:])
    #     current_state_vec = self._state_to_vector(state).reshape(1, -1)

    #     explainer = shap.KernelExplainer(self._predict_policy, background_data)
    #     with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    #         shap_values = explainer.shap_values(current_state_vec, nsamples=50)

    #     exploration_rate = f"Exploration rate: {self.current_policy.exploration:.3f}"

    #     explanation = f"Chọn hướng: {action_str}\n"
    #     explanation += f"Sheep Direction: {sheep_direction_str}\n"
    #     explanation += f"Facing Queue: {facing_queue_str if facing_queue_str else 'None'}\n"
    #     explanation += f"{exploration_rate}\n"
    #     explanation += "Giải thích SHAP:\n"
    #     explanation += f"- Sheep Direction: {shap_values[0][0]:+.2f}\n"
    #     explanation += f"- Facing Queue: {shap_values[0][1]:+.2f}\n"
    #     explanation += "\n" + self._interpret_shap_values(shap_values[0], state, action, current_direction)

    #     return explanation

    def explain_action(self, state, action, current_direction):
        """Giải thích hành động của người chăn cừu bằng SHAP """
        # Chuẩn bị thông tin cơ bản
        action_str = str(action).replace("Direction.", "")
        sheep_dir_str = str(state.sheep_direction if state.sheep_direction else "Không xác định").replace("ComplexDirection.", "")
        facing_queue_str = " ".join([str(d).replace("Direction.", "") for d in state.facing_queue]) or "Không có"

        # Kiểm tra dữ liệu lịch sử
        if not self.state_history:
            return (f"Người chăn cừu chọn hướng {action_str}.\n"
                    f"Chưa có đủ dữ liệu để giải thích chi tiết bằng SHAP.")

        # Chuyển trạng thái hiện tại thành vector
        current_state_vec = self._state_to_vector(state).reshape(1, -1)
        background_data = np.array(self.state_history[-100:])  # Dùng 100 mẫu gần nhất làm nền

        # Tính toán SHAP values, tắt output không cần thiết
        explainer = shap.KernelExplainer(self._predict_policy, background_data)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            shap_values = explainer.shap_values(current_state_vec, nsamples=50)[0]

        # Xây dựng giải thích tự nhiên
        exploration_rate = self.current_policy.exploration
        explanation = f"Người chăn cừu quyết định di chuyển theo hướng {action_str}. Lý do:\n\n"

        # Đánh giá ảnh hưởng của hướng cừu
        shap_sheep = shap_values[0]
        if abs(shap_sheep) > 0.1:
            if shap_sheep > 0:
                explanation += (f"- Hướng của cừu ({sheep_dir_str}) có tác động tích cực ({shap_sheep:+.2f}), "
                                f"khuyến khích người chăn cừu chọn hướng {action_str}.\n")
            else:
                explanation += (f"- Hướng của cừu ({sheep_dir_str}) có tác động tiêu cực ({shap_sheep:+.2f}), "
                                f"nhưng các yếu tố khác đã vượt qua để chọn hướng {action_str}.\n")
        else:
            explanation += (f"- Hướng của cừu ({sheep_dir_str}) không ảnh hưởng nhiều đến quyết định "
                            f"({shap_sheep:+.2f}).\n")

        # Đánh giá ảnh hưởng của hàng đợi chướng ngại
        shap_queue = shap_values[1]
        if abs(shap_queue) > 0.1:
            if shap_queue > 0:
                explanation += (f"- Các hướng bị chặn ({facing_queue_str}) hỗ trợ việc chọn hướng {action_str} "
                                f"(ảnh hưởng: {shap_queue:+.2f}).\n")
            else:
                explanation += (f"- Các hướng bị chặn ({facing_queue_str}) gây khó khăn cho hướng {action_str} "
                                f"(ảnh hưởng: {shap_queue:+.2f}), nhưng không đủ để thay đổi quyết định.\n")
        else:
            explanation += (f"- Các hướng bị chặn ({facing_queue_str}) hầu như không tác động "
                            f"({shap_queue:+.2f}).\n")

        # Thêm thông tin về hướng hiện tại và exploration rate
        current_dir_str = str(current_direction).replace("Direction.", "")
        explanation += (f"- Hướng hiện tại ({current_dir_str}) làm nền tảng cho quyết định này.\n"
                        f"- Tỷ lệ khám phá hiện tại: {exploration_rate:.3f}, cho thấy mức độ thử nghiệm trong lựa chọn.\n")

        # Tổng kết
        explanation += (f"\nKết luận: Quyết định chọn {action_str} dựa trên sự kết hợp giữa vị trí cừu, "
                        f"chướng ngại vật và hướng hiện tại, với các yếu tố được đánh giá qua SHAP.")

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