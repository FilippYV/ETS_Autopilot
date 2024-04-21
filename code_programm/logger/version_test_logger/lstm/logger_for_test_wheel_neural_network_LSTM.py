import time

import dxcam
import keyboard
import pygame
import torch
import torch.nn.functional as F
import vgamepad as vg
from ultralytics import YOLO

from code_programm.path import (get_path_config_road_area_size, get_path_config_speed_area_size,
                                get_path_weight_model)
from code_programm.wheel_neural_network.lstm.wheel_neural_network_lstm_3 import IS_LSTM_Net_v3


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


def get_weight_model():
    model_road = YOLO(get_path_weight_model('speed_recognition_v2.pt')).cuda()
    model_speed = YOLO(get_path_weight_model('speed_recognition.pt')).cuda()
    wheel_net = IS_LSTM_Net_v3().cuda()
    wheel_net.load_state_dict(torch.load(get_path_weight_model('weight_wheel_nn_lstm_3.pth')))
    return model_road, model_speed, wheel_net


def get_road_area_size():
    with open(get_path_config_road_area_size(), 'r') as file:
        lines = file.readlines()
    return [int(lines[1]), int(lines[1]) + int(lines[3]), int(lines[0]), int(lines[0]) + int(lines[2])]


def get_speed_area_size():
    with open(get_path_config_speed_area_size(), 'r') as file:
        lines = file.readlines()
    return [int(lines[1]), int(lines[1]) + int(lines[3]), int(lines[0]), int(lines[0]) + int(lines[2])]


def get_key_for_management():
    first_key = '\\'
    second_key = '\b'
    third_key = '-'
    fourth_key = "'"
    return first_key, second_key, third_key, fourth_key


def save_photo(camera, road_area, speed_area):
    screenshot_dxcam = camera.get_latest_frame()

    road = screenshot_dxcam[road_area[0]:road_area[1], road_area[2]:road_area[3]]
    speed = screenshot_dxcam[speed_area[0]:speed_area[1], speed_area[2]:speed_area[3]]

    return road, speed


def model_road_predict(model_road, road_img, combined_mask_old, combined_mask_new):
    prediction_road = model_road.predict(road_img,
                                         conf=0.6,
                                         verbose=False,
                                         device='cuda',
                                         show=False
                                         )

    if prediction_road[0].masks is not None:
        combined_mask_old = combined_mask_old * 0.4 + combined_mask_new * 0.6
        combined_mask_new.zero_()
        for i in prediction_road[0].masks.data:
            resized_mask = F.interpolate(i.unsqueeze(0).unsqueeze(0), size=(96, 128), mode='bilinear',
                                         align_corners=False)
            combined_mask_new += resized_mask.squeeze(0).unsqueeze(0)

    combined_tensor = combined_mask_old + combined_mask_new

    return combined_tensor.flatten().detach().unsqueeze(0), combined_mask_old, combined_mask_new


def model_speed_predict(model_speed, speed_img):
    speed_img
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
        speed = torch.tensor(int(speed), device='cuda').unsqueeze(0).unsqueeze(0)
    else:
        speed = torch.tensor(int(30), device='cuda').unsqueeze(0).unsqueeze(0)
    return speed


def smooth_transition(gamepad, start_position, end_position, steps=10):
    for step in range(steps):
        t = step / (steps - 1)  # t будет изменяться от 0 до 1
        current_position = (1 - t) * start_position + t * end_position
        gamepad.left_joystick_float(x_value_float=float(current_position), y_value_float=0.0)
        gamepad.update()
        time.sleep(0.01)  # Задержка для плавности перехода, может быть настроена


def new_position_gamepad(gamepad, wheel_position_mass):
    start_position = wheel_position_mass[0]  # Текущее положение
    end_position = wheel_position_mass[1]  # Желаемое положение

    # Плавный переход от start_position к end_position
    smooth_transition(gamepad, start_position, end_position)

    # Обновляем wheel_position_mass для следующего использования
    wheel_position_mass = torch.cat((wheel_position_mass[1:], wheel_position_mass[1:2]), 0)


# def new_position_gamepad(gamepad, wheel_position, wheel_position_mass):
#     wheel_position_mass = wheel_position_mass[0]
#     wheel_position_mass = torch.cat((wheel_position_mass, wheel_position), 0)
#     gamepad.left_joystick_float(x_value_float=wheel_position, y_value_float=0.0)
#     gamepad.update()


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

    camera = dxcam.create(device_idx=0, output_idx=0, output_color='BGR')
    camera.start(target_fps=60)

    mass_all = []

    opened = False
    recording = False

    try:
        while True:
            if recording:
                # counter = 0

                wheel_position_mass = torch.zeros([2], device='cuda')

                combined_mask_old = torch.zeros((1, 1, 96, 128), device='cuda')
                combined_mask_new = torch.zeros((1, 1, 96, 128), device='cuda')

                while opened:
                    start_time = time.time()

                    road_img, speed_img = save_photo(camera, road_area, speed_area)

                    combined_tensor, combined_mask_old, combined_mask_new = model_road_predict(
                        model_road, road_img, combined_mask_old, combined_mask_new)

                    speed = model_speed_predict(model_speed, speed_img)

                    print(combined_tensor.shape)
                    print(speed.shape)

                    input_tensor = torch.cat((combined_tensor, speed), dim=1)

                    wheel_position = model_wheel.forward(input_tensor)

                    # x = time.time() - start_time
                    # if x < 0.046:
                    #     time.sleep(0.046 - x)

                    new_position_gamepad(gamepad, wheel_position_mass)

                    # counter += 1

                    if keyboard.is_pressed(f'{first_key}'):
                        recording = False
                        opened = False

                        print('\nStop')

                        time.sleep(0.1)

                        print(sum(mass_all) / len(mass_all))

                    mass_all.append(time.time() - start_time)

            pygame.event.pump()
            gamepad.left_joystick_float(x_value_float=joystick.get_axis(0), y_value_float=0.0)
            gamepad.update()

            if keyboard.is_pressed(f'{first_key}'):
                recording = True
                opened = True

                print('\nStart')

                time.sleep(0.5)

            if keyboard.is_pressed(f'{third_key}'):
                print('\nCalibration')

                x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0]
                for xi in x:
                    gamepad.left_joystick_float(x_value_float=xi, y_value_float=0.0)
                    gamepad.update()
                    time.sleep(0.01)

            if keyboard.is_pressed(f'{second_key}'):
                print('\nExit')
                break

    except KeyboardInterrupt:
        pygame.quit()

    pygame.quit()


if __name__ == '__main__':
    main()
