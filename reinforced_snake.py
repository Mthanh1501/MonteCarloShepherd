import pygame
from montecarlo.brain import Brain
from montecarlo.state import State
from montecarlo.game.grid import Grid
from montecarlo.game.items import *
from random import choices

# Khởi tạo Pygame
pygame.init()

# Cửa sổ 1200x800: 800 cho trò chơi, 400 cho panel
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

def wrap_text(text, font, max_width):
    """Chia văn bản thành các dòng sao cho vừa với max_width."""
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
    # print("Gọi update_screen, explanation:", explanation if explanation else "Rỗng")
    # Vẽ panel giải thích
    pygame.draw.rect(canvas, (200, 200, 200), (800, 0, 400, 800))
    
    # Hiển thị giải thích
    if explanation:
        try:
            font = pygame.font.SysFont("timesnewroman", 24)
        except Exception as e:
            font = pygame.font.Font(None, 24)
        
        # Chia văn bản thành các dòng vừa với panel (rộng 400 pixel)
        max_width = 380  # Để lại lề 10 pixel mỗi bên
        lines = wrap_text(explanation, font, max_width)
        for i, line in enumerate(lines):
            text = font.render(line, True, (0, 0, 0))  # Màu đen
            canvas.blit(text, (810, 10 + i * 25))
    
    # Hiển thị thông báo tạm dừng nếu paused = True
    if paused:
        font = pygame.font.SysFont("timesnewroman", 36)
        pause_text = font.render("Đã tạm dừng - Nhấn Enter để tiếp tục", True, (255, 0, 0))  # Màu đỏ
        canvas.blit(pause_text, (200, 350))  # Giữa khu vực trò chơi
    
    pygame.display.update()
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

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            key_name = pygame.key.name(event.key)
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
                    # print("Explanation forced:", current_explanation)
                else:
                    current_explanation = ""
                    last_explanation = ""
            elif event.key == pygame.K_RETURN:
                paused = not paused
            elif manual and not paused:
                if event.key == pygame.K_a and direction != Direction.RIGHT:
                    direction = Direction.LEFT
                elif event.key == pygame.K_w and direction != Direction.DOWN:
                    direction = Direction.UP
                elif event.key == pygame.K_d and direction != Direction.LEFT:
                    direction = Direction.RIGHT
                elif event.key == pygame.K_s and direction != Direction.UP:
                    direction = Direction.DOWN

    # Chỉ cập nhật logic trò chơi khi không tạm dừng
    if not paused:
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
                    # print("Explanation updated (auto):", current_explanation)
        else:
            if show_explanation:
                new_explanation = "Chế độ thủ công: Dùng A/W/D/S để di chuyển"
                if new_explanation != last_explanation:
                    current_explanation = new_explanation
                    last_explanation = new_explanation
                    # print("Explanation updated (manual):", current_explanation)

        past_directions.insert(0, direction)
        if len(past_directions) >= 100:
            past_directions.pop()

        shepperd.move(direction)

        if len(past_positions) >= 200:
            past_positions.pop()

        if shepperd.x_cell == current_sheep.x_cell and shepperd.y_cell == current_sheep.y_cell:
            current_sheep = Sheep(grid.random_cell(), grid.random_cell())
            shepperd.sheeps += 1
            brain.add_reward(50)
        else:
            brain.add_reward(-1)

        for i in range(shepperd.sheeps):
            if (shepperd.x_cell, shepperd.y_cell) == past_positions[i]:
                shepperd = Shepperd(0, 5, grid)
                current_sheep = Sheep(grid.random_cell(), grid.random_cell())
                brain.add_reward(-300)
                brain.evaluate()
                break

        past_positions.insert(0, (shepperd.x_cell, shepperd.y_cell))

        if print_policy:
            print(brain.current_policy)

    # Vẽ giao diện
    canvas.fill((255, 255, 255))  # Nền trắng cho khu vực trò chơi
    canvas.blit(shepperd_sprite, (grid.from_cell(shepperd.x_cell), grid.from_cell(shepperd.y_cell)))
    for i in range(0, shepperd.sheeps):
        canvas.blit(cheese_sprite, (grid.from_cell(past_positions[i + 1][0]), grid.from_cell(past_positions[i + 1][1])))
    canvas.blit(sheep_sprite, (grid.from_cell(current_sheep.x_cell), grid.from_cell(current_sheep.y_cell)))

    # Cập nhật màn hình
    update_screen(current_explanation, paused)

pygame.quit()