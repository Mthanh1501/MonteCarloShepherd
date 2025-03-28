import pygame
import matplotlib.pyplot as plt
from montecarlo.brain import Brain
from montecarlo.state import State
from montecarlo.game.grid import Grid
from montecarlo.game.items import *
from random import choices

# Khởi tạo Pygame
pygame.init()

canvas = pygame.display.set_mode((1200, 800))
pygame.display.set_caption("Shepherd Game with Explanation")

# Tải sprite
try:
    sheep_sprite = pygame.image.load("sprites/Sheep.jpg")
    shepperd_sprite = pygame.image.load("sprites/Shepperd.jpg")
    cheese_sprite = pygame.image.load("sprites/Cheese.jpg")
    shepperd_sprite = pygame.transform.scale(shepperd_sprite, (50, 50))
    sheep_sprite = pygame.transform.scale(sheep_sprite, (50, 50))
    cheese_sprite = pygame.transform.scale(cheese_sprite, (50, 50))
except pygame.error as e:
    pygame.quit()
    exit()

FPS = 10

# Biến để kiểm soát biểu đồ
plot_shown = False

# Danh sách lưu số bước khi bắt được cừu
steps_to_reward = []

def show_plot(steps_data):
    """Hiển thị biểu đồ với trục x là lần bắt được cừu và trục y là số bước"""
    global plot_shown
    if not plot_shown:
        plt.ion()  # Bật chế độ tương tác
        plt.figure(figsize=(8, 5))  # Tạo figure mới nếu chưa có
        plot_shown = True
    else:
        plt.clf()  # Xóa nội dung biểu đồ cũ
    
    # Vẽ biểu đồ mới
    plt.xlabel("Lần bắt được cừu")
    plt.ylabel("Số bước")
    plt.title("Số bước để bắt được cừu qua từng lần")
    plt.grid(True)
    
    plt.plot(range(len(steps_data)), steps_data, label="Steps", color='purple', marker='o', linestyle='None')  # Chỉ hiển thị điểm
    plt.xlim(0, len(steps_data) - 1 if steps_data else 1)  # Giới hạn trục x dựa trên số lần
    plt.ylim(0, max(steps_data) if steps_data else 1)  # Giới hạn trục y dựa trên số bước tối đa
    plt.legend()
    
    plt.draw()  # Cập nhật biểu đồ
    plt.pause(0.001)  # Đảm bảo biểu đồ được hiển thị ngay lập tức

def hide_plot():
    """Ẩn biểu đồ"""
    global plot_shown
    if plot_shown:
        plt.close()
        plot_shown = False

def wrap_text(text, font, max_width):
    lines = []
    for line in text.split('\n'):
        words = line.split(' ')
        current_line = ''
        for word in words:
            test_line = current_line + word + ' '
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + ' '
        if current_line:
            lines.append(current_line.strip())
    return lines

def update_screen(explanation="", paused=False):
    pygame.draw.rect(canvas, (200, 200, 200), (800, 0, 400, 800))
    if explanation:
        try:
            font = pygame.font.SysFont("timesnewroman", 24)
        except Exception as e:
            font = pygame.font.Font(None, 24)
        max_width = 380
        lines = wrap_text(explanation, font, max_width)
        for i, line in enumerate(lines):
            text = font.render(line, True, (0, 0, 0))
            canvas.blit(text, (810, 10 + i * 25))
    if paused:
        font = pygame.font.SysFont("timesnewroman", 36)
        pause_text = font.render("Đã tạm dừng - Nhấn Enter để tiếp tục", True, (255, 0, 0))
        canvas.blit(pause_text, (200, 350))
    pygame.display.flip()
    pygame.time.Clock().tick(FPS)

grid = Grid(16, 800)
shepperd = Shepperd(0, 5, grid)
current_sheep = Sheep(grid.random_cell(), grid.random_cell())
brain = Brain(gamma=0.78)

running = True
paused = False
print_state = False
print_policy = False
manual = False
show_explanation = False

past_positions = []
past_directions = []
direction = Direction.RIGHT
current_explanation = ""
last_explanation = ""

# Biến đếm bước
step_count = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                FPS = 15000 if FPS == 10 else 10
            elif event.key == pygame.K_f:
                print_state = not print_state
            elif event.key == pygame.K_p:
                print_policy = not print_policy
            elif event.key == pygame.K_m:
                manual = not manual
            elif event.key == pygame.K_e:
                show_explanation = not show_explanation
                if show_explanation:
                    current_explanation = "Test: Giải thích đã bật\nNhấn E để tắt"
                else:
                    current_explanation = ""
                    last_explanation = ""
            elif event.key == pygame.K_RETURN:
                paused = not paused  # Enter chỉ tạm dừng hoặc tiếp tục
                if not paused:
                    hide_plot()  # Ẩn biểu đồ khi tiếp tục chơi
            elif event.key == pygame.K_q:
                if not paused:  # Nếu chưa tạm dừng, nhấn Q sẽ tạm dừng
                    paused = True
                show_plot(steps_to_reward)  # Hiển thị biểu đồ khi nhấn Q
            elif manual and not paused:
                if event.key == pygame.K_a and direction != Direction.RIGHT:
                    direction = Direction.LEFT
                elif event.key == pygame.K_w and direction != Direction.DOWN:
                    direction = Direction.UP
                elif event.key == pygame.K_d and direction != Direction.LEFT:
                    direction = Direction.RIGHT
                elif event.key == pygame.K_s and direction != Direction.UP:
                    direction = Direction.DOWN

    if not paused:
        step_count += 1  # Tăng số bước mỗi lần di chuyển

        if not manual:
            state = State(shepperd.get_sheep_direction(current_sheep), shepperd.get_queue_directions(past_positions[1:shepperd.sheeps], direction))
            if print_state:
                print(state)
            direction = brain.choose_direction(state, direction)
            if show_explanation:
                new_explanation = brain.explain_action(state, direction, direction)
                if new_explanation != last_explanation:
                    current_explanation = new_explanation
                    last_explanation = new_explanation
        else:
            if show_explanation:
                new_explanation = "Chế độ thủ công: Dùng A/W/D/S để di chuyển"
                if new_explanation != last_explanation:
                    current_explanation = new_explanation
                    last_explanation = new_explanation

        past_directions.insert(0, direction)
        if len(past_directions) >= 100:
            past_directions.pop()

        shepperd.move(direction)

        if len(past_positions) >= 200:
            past_positions.pop()

        if shepperd.x_cell == current_sheep.x_cell and shepperd.y_cell == current_sheep.y_cell:
            # Khi bắt được cừu, lưu số bước
            current_sheep = Sheep(grid.random_cell(), grid.random_cell())
            shepperd.sheeps += 1
            brain.add_reward(50)
            steps_to_reward.append(step_count)
            # Reset lại số bước
            step_count = 0
        else:
            brain.add_reward(-1)

        for i in range(shepperd.sheeps):
            if (shepperd.x_cell, shepperd.y_cell) == past_positions[i]:
                shepperd = Shepperd(0, 5, grid)
                current_sheep = Sheep(grid.random_cell(), grid.random_cell())
                brain.add_reward(-300)
                brain.evaluate()
                # Reset lại số bước khi va chạm
                step_count = 0
                break

        past_positions.insert(0, (shepperd.x_cell, shepperd.y_cell))

        if print_policy:
            print(brain.current_policy)

    canvas.fill((255, 255, 255))
    canvas.blit(shepperd_sprite, (grid.from_cell(shepperd.x_cell), grid.from_cell(shepperd.y_cell)))
    for i in range(0, shepperd.sheeps):
        canvas.blit(cheese_sprite, (grid.from_cell(past_positions[i + 1][0]), grid.from_cell(past_positions[i + 1][1])))
    canvas.blit(sheep_sprite, (grid.from_cell(current_sheep.x_cell), grid.from_cell(current_sheep.y_cell)))

    update_screen(current_explanation, paused)

pygame.quit()
hide_plot()  # Đảm bảo đóng biểu đồ khi thoát