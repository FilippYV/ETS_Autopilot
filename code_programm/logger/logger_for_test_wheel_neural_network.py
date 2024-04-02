import time

import cv2
import keyboard
import numpy as np
import pygame
import torch
import vgamepad as vg
from PIL import ImageGrab
from ultralytics import YOLO

from code_programm.path import (get_path_config_road_area_size, get_path_config_speed_area_size,
                                get_path_weight_model)
from code_programm.wheel_neural_network.wheel_net import FullyConnectedNN


def get_weight_model():
    model_road = YOLO(get_path_weight_model('line_recognition.pt'))
    model_speed = YOLO(get_path_weight_model('speed_recognition.pt'))
    wheel_net = FullyConnectedNN().cuda()
    wheel_net.load_state_dict(torch.load(get_path_weight_model('weight_wheel_nn.pth')))
    return model_road, model_speed, wheel_net


def get_road_area_size():
    with open(get_path_config_road_area_size(), 'r') as file:
        lines = file.readlines()
    return [int(lines[0]), int(lines[1]), int(lines[2]), int(lines[3])]


def get_speed_area_size():
    with open(get_path_config_speed_area_size(), 'r') as file:
        lines = file.readlines()
    return [int(lines[0]), int(lines[1]), int(lines[2]), int(lines[3])]


def get_key_for_management():
    first_key = '\\'
    second_key = '\b'
    third_key = '-'
    fourth_key = "'"
    return first_key, second_key, third_key, fourth_key


def save_photo(road_area, speed_area):
    screenshot = ImageGrab.grab()
    road = screenshot.crop((road_area[0],
                            road_area[1],
                            road_area[0] + road_area[2],
                            road_area[1] + road_area[3]))
    road_resize = cv2.resize(np.array(road), (512, 512))
    cv2.imshow('Road', road_resize)
    speed = screenshot.crop((speed_area[0],
                             speed_area[1],
                             speed_area[0] + speed_area[2],
                             speed_area[1] + speed_area[3]))
    return road_resize, speed


def model_road_predict(model_road, road_img, combined_mask):
    prediction_road = model_road.predict(road_img, imgsz=512, conf=0.3, verbose=False, device='cuda', show=False)

    combined_mask *= 0

    if prediction_road[0].masks is not None:
        for i in prediction_road[0].masks.data:
            combined_mask += i.cpu().numpy()

    images = cv2.resize(combined_mask.copy(), (128, 128))

    return images.flatten(), combined_mask

def model_speed_predict(model_speed, speed_img):
    results = model_speed.predict(speed_img, conf=0.9, device='cuda', verbose=False, show=False)
    sorted_objects = sorted(
        ({'class': int(cls), 'confidence': float(conf), 'xmin': int(xmin), 'ymin': int(ymin), 'xmax': int(xmax),
          'ymax': int(ymax)}
         for result in results for obj in result.boxes.data for xmin, ymin, xmax, ymax, conf, cls in
         (obj.tolist(),)),
        key=lambda obj: obj['xmin']
    )
    if sorted_objects:
        speed = ''.join(str(obj['class']) for obj in sorted_objects)
    else:
        speed = '0'
    return speed


def new_position_gamepad(gamepad, wheel_position):
    gamepad.left_joystick_float(x_value_float=wheel_position.item(), y_value_float=0.0)
    gamepad.update()


def start_work_pygame():
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


def start_work_gamepad():
    gamepad = vg.VX360Gamepad()
    gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    gamepad.update()
    return gamepad


def main():
    pygame.init()

    # Инициализация джойстика
    joystick = start_work_pygame()
    gamepad = start_work_gamepad()

    # Получаем области для распознавания
    road_area = get_road_area_size()
    speed_area = get_speed_area_size()

    # Объявляем нейросети и веса
    model_road, model_speed, model_wheel = get_weight_model()

    # Объявляем кнопки для управления
    first_key, second_key, third_key, fourth_key = get_key_for_management()

    combined_mask = np.zeros((road_area[2], road_area[2]))

    mass_all = []

    opened = False
    recording = False

    try:
        while True:
            if recording:
                counter = 0
                while opened:
                    start_time = time.time()

                    road_img, speed_img = save_photo(road_area, speed_area)

                    pygame.event.pump()

                    images, combined_mask = model_road_predict(model_road, road_img, combined_mask)

                    speed = model_speed_predict(model_speed, speed_img)

                    combined_mask_ = np.append(images, speed)

                    tensor_combined_mask = torch.tensor(combined_mask_.astype(np.float32),
                                                        dtype=torch.float32).cuda()

                    wheel_position = model_wheel.forward(tensor_combined_mask)

                    new_position_gamepad(gamepad, wheel_position)

                    counter += 1

                    if keyboard.is_pressed(f'{first_key}'):
                        recording = False
                        opened = False

                        print('\nStop')

                        time.sleep(0.5)

                        print(sum(mass_all) / len(mass_all))

                    mass_all.append(time.time() - start_time)

            pygame.event.pump()
            gamepad.left_joystick_float(x_value_float=joystick.get_axis(0), y_value_float=0.0)
            gamepad.update()

            if keyboard.is_pressed(f'{first_key}'):
                recording = True
                opened = True

                print('\nStart')

                time.sleep(1)
            if keyboard.is_pressed(f'{third_key}'):
                print('\nCalibration')

                x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0]
                for xi in x:
                    gamepad.left_joystick_float(x_value_float=xi, y_value_float=0.0)
                    gamepad.update()
                    time.sleep(0.01)

            if keyboard.is_pressed(f'{second_key}'):
                recording = False
                opened = False

                print('\nExit')
                break

    except KeyboardInterrupt:
        pygame.quit()


pygame.quit()

if __name__ == '__main__':
    main()
