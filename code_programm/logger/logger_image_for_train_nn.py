import time

import keyboard
import pyautogui
import pygame

from code_programm.path import get_path_save_screens, get_path_name, get_path_save_csv

pygame.init()
joystick_count = pygame.joystick.get_count()
if joystick_count > 0:
    try:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(F"Гей\мпад найден: {joystick.get_name()}")
    except KeyError:
        print("Геймпад не найден.")
        pygame.quit()
        quit()

# Для скриншота по области
left = 710
top = 230
width = 512
height = 512
counter = 0

first_key = ']'
second_key = '\\'

num_axes = joystick.get_numaxes()

step = 3
stop_time = 0

S = False

recording = False
opened = False
while True:
    try:
        if recording:
            with open(filename, 'w') as file:
                while opened:
                    if recording:
                        pygame.event.pump()
                        if not S:
                            name_screens = time.time()
                            stop_time = name_screens + step
                            S = True
                        if name_screens - stop_time >= 0:
                            file.write(f'{name_screens},{joystick.get_axis(0)}\n')
                            screenshot = pyautogui.screenshot(region=(left, top, width, height))
                            screenshot.save(get_path_save_screens(f'{name_path}', f'{name_screens}.png'))
                            print(f'{counter}')
                            counter += 1
                            S = False
                        else:
                            name_screens = time.time()

                    if keyboard.is_pressed(f'{first_key}'):
                        recording = False
                        opened = False
                        file.close()
                        print('second')
                        name_screens = 0
                        counter = 0
                        time.sleep(1)

        if keyboard.is_pressed(f'{first_key}'):
            recording = True
            opened = True

            path, name_path = get_path_name()
            filename = get_path_save_csv(path)
            print('fist')
            print('filename')
            time.sleep(1)

        if keyboard.is_pressed(f'{second_key}'):
            recording = False
            opened = False
            file.close()
            print('three')
            break

    except KeyboardInterrupt:
        pygame.quit()
