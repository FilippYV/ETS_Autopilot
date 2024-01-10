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
        print(F"Геймпад найден: {joystick.get_name()}")
    except KeyError:
        print("Геймпад не найден.")
        pygame.quit()
        quit()

num_axes = joystick.get_numaxes()

name_screens = 0

recording = False
opened = False
while True:
    try:
        if recording:
            with open(filename, 'w') as file:
                while opened:
                    if recording:
                        pygame.event.pump()
                        file.write(f'{name_screens},{joystick.get_axis(0)}\n')
                        screenshot = pyautogui.screenshot()
                        screenshot.save(get_path_save_screens(f'{name_path}', f'{name_screens}.png'))
                        name_screens += 1
                        time.sleep(1 / 15)

                    if keyboard.is_pressed('right alt'):
                        recording = False
                        opened = False
                        file.close()
                        print('second')
                        name_screens = 0
                        time.sleep(1)

        if keyboard.is_pressed('right alt'):
            recording = True
            opened = True

            path, name_path = get_path_name()
            filename = get_path_save_csv(path)
            print('fist')
            time.sleep(1)


    except KeyboardInterrupt:
        pygame.quit()
