import time

import os
import cv2
import dxcam
import keyboard
import pygame

from code_programm.path import (get_path_config_road_area_size, get_path_config_speed_area_size,
                                get_path_save_screens_road, get_path_save_screens_speed, get_path_name,
                                get_path_save_csv)


def get_road_area_size():
    with open(get_path_config_road_area_size(), 'r') as file:
        lines = file.readlines()
    return [int(lines[1]), int(lines[1]) + int(lines[3]), int(lines[0]), int(lines[0]) + int(lines[2])]


def get_speed_area_size():
    with open(get_path_config_speed_area_size(), 'r') as file:
        lines = file.readlines()
    return [int(lines[1]), int(lines[1]) + int(lines[3]), int(lines[0]), int(lines[0]) + int(lines[2])]


def create_path(path):
    path_road = os.path.join(path, 'road')
    path_speed = os.path.join(path, 'speed')
    if not os.path.exists(path_road):
        os.makedirs(path_road)
        print(f"Папка '{path_road}' была создана.")
    else:
        print(f"Папка '{path_road}' уже существует.")
    if not os.path.exists(path_speed):
        os.makedirs(path_speed)
        print(f"Папка '{path_speed}' была создана.")
    else:
        print(f"Папка '{path_speed}' уже существует.")
    return path_road, path_speed


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


def save_photo(camera, road_area, speed_area):
    screenshot_dxcam = camera.get_latest_frame()
    road = screenshot_dxcam[road_area[0]:road_area[1], road_area[2]:road_area[3]]

    speed = screenshot_dxcam[speed_area[0]:speed_area[1], speed_area[2]:speed_area[3]]

    return road, speed


def main():
    # Получаем области для распознавания
    road_area = get_road_area_size()
    speed_area = get_speed_area_size()

    # Инициализация джойстика
    joystick = start_work_pygame()
    first_key, second_key = get_key_for_management()

    camera = dxcam.create(device_idx=0, output_idx=0, output_color='BGR')
    camera.start(target_fps=60)

    opened = False
    recording = False
    step = 0.033

    log_file_path = ''
    name_path = ''
    path = ''

    while True:
        try:
            if recording:
                with open(log_file_path, 'w') as file:
                    path_road, path_speed = create_path(path)
                    counter = 0
                    while opened:
                        start_time = time.time()

                        road, speed = save_photo(camera, road_area, speed_area)

                        pygame.event.pump()
                        wheel_position = joystick.get_axis(0)

                        file.write(f'{name_path}_{counter},{wheel_position}\n')

                        cv2.imwrite(os.path.join(rf'{path_road}', f'{name_path}_{counter}.png'), road)
                        cv2.imwrite(os.path.join(rf'{path_speed}', f'{name_path}_{counter}.png'), speed)

                        counter += 1

                        time_program = step - (time.time() - start_time)
                        if time_program > 0:
                            time.sleep(time_program)

                        if keyboard.is_pressed(f'{first_key}'):
                            print('\nStop')

                            recording = False
                            opened = False

                            file.close()

                            time.sleep(1)

            if keyboard.is_pressed(f'{first_key}'):
                recording = True
                opened = True

                path, name_path = get_path_name()
                log_file_path = get_path_save_csv(path)

                print('\nStart')

                time.sleep(0.5)

                print(f'{name_path}')

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
