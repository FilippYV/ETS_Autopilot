import time

import keyboard
import pygame
import cv2
import numpy as np
from PIL import ImageGrab

from code_programm.path import (get_path_config_road_area_size, get_path_config_speed_area_size,
                                get_path_save_screens_road, get_path_save_screens_speed, get_path_name,
                                get_path_save_csv)


def get_road_area_size():
    with open(get_path_config_road_area_size(), 'r') as file:
        lines = file.readlines()
    return [int(lines[0]), int(lines[1]), int(lines[2]), int(lines[3])]


def get_speed_area_size():
    with open(get_path_config_speed_area_size(), 'r') as file:
        lines = file.readlines()
    return [int(lines[0]), int(lines[1]), int(lines[2]), int(lines[3])]


def start_work_pygame():
    pygame.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(F"Геймпад найден: {joystick.get_name()}")
        return joystick
    else:
        print("Геймпад не найден.")
        pygame.quit()
        quit()


def get_key_for_management():
    first_key = '\\'
    second_key = '\b'
    return first_key, second_key


def save_photo(road_area, speed_area):
    screenshot = ImageGrab.grab()
    road = screenshot.crop((road_area[0],
                            road_area[1],
                            road_area[0] + road_area[2],
                            road_area[1] + road_area[3]))

    speed = screenshot.crop((speed_area[0],
                             speed_area[1],
                             speed_area[0] + speed_area[2],
                             speed_area[1] + speed_area[3]))

    return road, speed


def main():
    # Получаем области для распознавания
    road_area = get_road_area_size()
    speed_area = get_speed_area_size()

    # Инициализация джойстика
    joystick = start_work_pygame()
    first_key, second_key = get_key_for_management()

    opened = False
    recording = False
    step = 0.07

    log_file_path = ''
    name_path = ''

    while True:
        try:
            if recording:
                with open(log_file_path, 'w') as file:
                    counter = 0
                    times = False
                    while opened:
                        if not times:
                            start_time = time.time()

                            road, speed = save_photo(road_area, speed_area)

                            pygame.event.pump()
                            wheel_position = joystick.get_axis(0)

                            file.write(f'{name_path}_{counter},{wheel_position}\n')

                            road.save(
                                get_path_save_screens_road(rf'{name_path}', f'{name_path}_{counter}.png')
                            )
                            speed.save(
                                get_path_save_screens_speed(rf'{name_path}', f'{name_path}_{counter}.png')
                            )

                            counter += 1

                        time_program = step - (time.time() - start_time)
                        if time_program > 0:
                            time.sleep(time_program)

                        if keyboard.is_pressed(f'{first_key}'):
                            recording = False
                            opened = False

                            file.close()

                            print('\nStop')

                            time.sleep(1)

            if keyboard.is_pressed(f'{first_key}'):
                recording = True
                opened = True

                path, name_path = get_path_name()
                log_file_path = get_path_save_csv(path)

                print('\nStart')
                print(f'{name_path}')

                time.sleep(1)

            if keyboard.is_pressed(f'{second_key}'):
                recording = False
                opened = False

                file.close()

                print('\nExit')
                break

        except KeyboardInterrupt:
            pygame.quit()


if __name__ == '__main__':
    main()
