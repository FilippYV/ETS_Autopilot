import time

import cv2
import keyboard
import numpy as np
import pyautogui
import pygame
import torch
import vgamepad as vg
from ultralytics import YOLO

from code_programm.path import get_path_name, get_path_save_csv
from code_programm.path import get_path_weight_model
from code_programm.path import get_path_wheel_net_weight
from code_programm.whel_neural_network.wheel_net import FullyConnectedNN

wheel_net = FullyConnectedNN().cuda()
wheel_net.load_state_dict(torch.load(get_path_wheel_net_weight()))

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
gamepad = vg.VX360Gamepad()

gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
gamepad.update()


def set_wheel_axis(value):
    # Преобразование значения в диапазон от -1 до 1
    normalized_value = max(-1, min(1, value))
    # Масштабирование значения к диапазону 0-65535 (максимальное значение для оси)
    scaled_value = int((normalized_value + 1) * 32767)

    gamepad.set_vid(vg.XUSB_AXIS.RX, scaled_value)
    gamepad.update()


print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
model = YOLO(get_path_weight_model('best_v2_n.pt'))

# Для скриншота по области
left = 710
top = 230
width = 512
height = 512

combined_mask = np.zeros((512, 512))
counter = 0
first_key = ']'
second_key = '\\'

step = 0.05
stop_time = 0

S = False

recording = False
opened = False
while True:
    try:
        if recording:
            while opened:
                if recording:
                    pygame.event.pump()
                    if not S:
                        name_screens = time.time()
                        stop_time = name_screens + step
                        S = True
                    if name_screens - stop_time >= 0:
                        start_time = time.time()
                        screenshot = pyautogui.screenshot(region=(left, top, width, height))
                        results = model(screenshot, imgsz=512, conf=0.65, show=False, device='cuda')

                        combined_mask -= combined_mask * 0.25

                        if results[0].masks is not None:
                            for i in results[0].masks.data:
                                combined_mask += i.cpu().numpy()
                        images = cv2.resize(combined_mask.copy(), (32, 128))
                        print(len(images), images.size, len(images.flatten()))
                        new_image = torch.tensor(images.flatten(), dtype=torch.float32).cuda()
                        # print(f'new_image, {len(new_image)}')
                        angle = wheel_net.forward(new_image).item()
                        print(f'result_nn: {angle}')
                        print(joystick.get_axis(0))
                        gamepad.left_joystick_float(x_value_float=round(angle, 2), y_value_float=0.0)
                        gamepad.update()
                        counter += 1
                        S = False
                        print('Время', time.time() - start_time)
                    else:
                        name_screens = time.time()

                if keyboard.is_pressed(f'{first_key}'):
                    recording = False
                    opened = False
                    print('Stop')
                    name_screens = 0
                    counter = 0
                    gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
                    gamepad.update()
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
            print('three')
            gamepad.stop()
            break

        if keyboard.is_pressed(f'['):
            gamepad.left_joystick_float(x_value_float=0.5, y_value_float=0.0)
            gamepad.update()


    except KeyboardInterrupt:
        pygame.quit()
