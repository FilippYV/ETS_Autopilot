import time

import keyboard
import dxcam
import cv2
import numpy as np
import time


camera = dxcam.create(device_idx=0, output_idx=0, output_color='BGR')
camera.start()
x = True
y = False

while x:
    while y:
        screenshot_dxcam = camera.get_latest_frame()
        road = screenshot_dxcam[376:377, 360:361]
        files = np.array(road)
        image = files.flatten()
        if image[0] == image[1] == image[2]:
            keyboard.press_and_release('ENTER')
            print('Stop\n')
            y = False

        if keyboard.is_pressed('\\'):
            y = False
            print('False')
            time.sleep(0.1)

    if keyboard.is_pressed('\\'):
        y = True
        print('Start')
        time.sleep(0.1)

    if keyboard.is_pressed(']'):
        x = False
        print('End')
        time.sleep(0.1)
