import pygame
import sys

# Инициализация Pygame
pygame.init()

# Установка размеров экрана
screen_width = 1280
screen_height = 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Контур квадрата")

# Параметры квадрата
left = 710
top = 230
width = 512
height = 512
line_width = 2
square_color = (255, 255, 255)  # Белый цвет

# Основной цикл программы
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Заливка экрана черным цветом
    screen.fill((0, 0, 0))

    # Рисование контура квадрата
    pygame.draw.rect(screen, square_color, (left, top, width, height), line_width)

    # Обновление экрана
    pygame.display.flip()

# Выход из программы
pygame.quit()
sys.exit()