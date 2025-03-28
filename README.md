## ReinforcedSnake – Học Tăng Cường với XRL trong Trò Chơi Rắn Săn Mồi
# Giới thiệu

ReinforcedSnake là một trò chơi Rắn săn mồi được triển khai bằng PyGame, sử dụng học tăng cường (Reinforcement Learning - RL) kết hợp với Học Tăng Cường Giải Thích được (Explainable Reinforcement Learning - XRL).

Chương trình áp dụng Monte Carlo Control để điều chỉnh chính sách di chuyển của rắn, giúp nó tự học cách né tránh chính mình và tối ưu hóa hành vi săn thức ăn. Khi tần số khám phá (exploration rate) giảm về 0, rắn sẽ chơi hoàn hảo bằng cách sử dụng chính sách đã học được.

XRL giúp cung cấp giải thích về hành động, giúp người chơi hiểu rõ hơn tại sao rắn chọn một hướng di chuyển nhất định.

# Các thành phần chính
Chương trình bao gồm các thành phần chính sau:

    Evaluation – Đánh giá trạng thái và tính toán phần thưởng.
    Policy – Xây dựng chính sách di chuyển tối ưu dựa trên học tăng cường.
    State – Biểu diễn trạng thái trò chơi, bao gồm vị trí rắn, hướng di chuyển và thức ăn.

Explainable RL (XRL) – Cung cấp thông tin giải thích chi tiết về quyết định của rắn, giúp người dùng hiểu rõ cách thuật toán hoạt động.

Cách điều khiển:

    Phím Space – Tăng tốc độ FPS và tốc độ học.
    Phím Enter – Tạm dừng trò chơi.
    Phím M – Kích hoạt chế độ tự động chơi.
    Phím E – Xem giải thích chi tiết về lý do rắn chọn hướng di chuyển.
