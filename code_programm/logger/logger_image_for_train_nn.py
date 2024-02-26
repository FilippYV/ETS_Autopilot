import time

import keyboard
from PIL import ImageGrab

from code_programm.path import get_path_save_screens, get_path_name, get_path_save_csv

# Для скриншота по области
left = 710
top = 230
width = 512
height = 512

interval_screens = 1

first_key = '/'
second_key = '.'

recording = False
opened = False
name_screens = False

while True:
    if recording:
        with open(filename, 'w') as file:
            while opened:
                if recording:
                    if abs(name_screens - time.time()) > interval_screens or name_screens is False:
                        name_screens = time.time()
                        screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
                        screenshot.save(get_path_save_screens(f'{name_path}', f'{name_screens}.png'))

                if keyboard.is_pressed(f'{first_key}'):
                    recording = False
                    opened = False
                    file.close()
                    print('second')
                    time.sleep(1)

    if keyboard.is_pressed(f'{first_key}'):
        recording = True
        opened = True

        path, name_path = get_path_name()
        filename = get_path_save_csv(path)
        print('fist')
        time.sleep(1)

    if keyboard.is_pressed(f'{second_key}'):
        recording = False
        opened = False
        file.close()
        print('three')
        break
