import time
from queue import Queue
from threading import Thread, Event

import cv2
import dxcam
import keyboard
import numpy as np
import pygame
import torch
import torch.nn.functional as F
import vgamepad as vg
from ultralytics import YOLO

from code_programm.path import (get_path_config_road_area_size, get_path_config_speed_area_size,
                                get_path_weight_model)
from code_programm.wheel_neural_network.forward.ver_5.wheel_neural_network_forward_5 import FeedforwardNet


def start_work_wheel_pygame():
    joystick_count = pygame.joystick.get_count()
    if joystick_count > 0:
        joystick = pygame.joystick.Joystick(joystick_count - 1)
        joystick.init()
        print(F"Геймпад найден: {joystick.get_name()}")
        return joystick
    else:
        print("Геймпад не найден.")
        pygame.quit()
        quit()


def start_work_virtual_gamepad():
    gamepad = vg.VX360Gamepad()
    gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    gamepad.update()
    return gamepad


def get_road_area_size():
    with open(get_path_config_road_area_size(), 'r') as file:
        lines = file.readlines()
    return [int(lines[1]), int(lines[1]) + int(lines[3]), int(lines[0]), int(lines[0]) + int(lines[2])]


def get_speed_area_size():
    with open(get_path_config_speed_area_size(), 'r') as file:
        lines = file.readlines()
    return [int(lines[1]), int(lines[1]) + int(lines[3]), int(lines[0]), int(lines[0]) + int(lines[2])]


def get_weight_model():
    model_road = YOLO(get_path_weight_model('best.pt')).cuda()
    model_speed = YOLO(get_path_weight_model('speed_recognition.pt')).cuda()
    wheel_net = FeedforwardNet().cuda()
    wheel_net.load_state_dict(torch.load(get_path_weight_model('weight_wheel_nn_forward_5.pth')))
    return model_road, model_speed, wheel_net


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


def model_road_processing(model_road, road_image, combined_mask_old, combined_mask_new):
    result_model_road = model_road(road_image,
                                   conf=0.3,
                                   device='cuda',
                                   verbose=False,
                                   # show=True
                                   )
    if result_model_road[0].masks is not None:
        combined_mask_old = combined_mask_new
        combined_mask_new.zero_()
        for i in result_model_road[0].masks.data:
            resized_mask = F.interpolate(i.unsqueeze(0).unsqueeze(0), size=(96, 128), mode='bilinear',
                                         align_corners=False)
            combined_mask_new += resized_mask.squeeze(0).unsqueeze(0)

    combined_tensor = combined_mask_old + combined_mask_new

    return combined_tensor.flatten().detach(), combined_mask_old, combined_mask_new


def model_speed_processing(model_speed, speed_image):
    result_model_speed = model_speed(speed_image,
                                     conf=0.9,
                                     device='cuda',
                                     verbose=False,
                                     # show=True
                                     )
    sorted_objects = sorted(
        ({'class': int(cls), 'confidence': float(conf), 'xmin': int(xmin), 'ymin': int(ymin), 'xmax': int(xmax),
          'ymax': int(ymax)}
         for result in result_model_speed for obj in result.boxes.data for xmin, ymin, xmax, ymax, conf, cls in
         (obj.tolist(),)),
        key=lambda obj: obj['xmin']
    )
    if sorted_objects:
        speed = int(''.join(str(obj['class']) for obj in sorted_objects))
        speed = torch.tensor([speed//170], device='cuda')
    else:
        speed = torch.tensor([30//170], device='cuda')
    return speed


def update_wheel_position_from_queue(virtual_wheel, queue_speed_pos, last_wheel_position):
    while True:
        positions = queue_speed_pos.get()
        if positions != last_wheel_position:
            for position in positions:
                virtual_wheel.left_joystick_float(x_value_float=position, y_value_float=0.0)
                virtual_wheel.update()
                time.sleep(0.000001)
            last_wheel_position = positions


def linear_interpolation(v0, v1, t):
    return v0 + (v1 - v0) * t


def interpolation_wheel_position(new_array_wheel_position, position_gamepad_value, array_wheel_pos):
    new_value = position_gamepad_value[1].item()
    old_value = new_array_wheel_position[-1]
    delta_values = np.abs(new_value - old_value)
    delta = 1

    t_values = np.array(array_wheel_pos) / delta
    interpolated_values = old_value + (new_value - old_value) * t_values

    return interpolated_values.tolist()



def clear_queue(q):
    with q.mutex:
        unused_items = q.queue.clear()


def update_virtual_wheel(user_wheel_update, virtual_wheel_update, stop_event):
    while not stop_event.is_set():
        pygame.event.pump()
        virtual_wheel_update.left_joystick_float(x_value_float=user_wheel_update.get_axis(0), y_value_float=0.0)
        virtual_wheel_update.update()
        time.sleep(0.000001)


def main_code():
    # Инициализация PyGame
    pygame.init()

    # Инициализация джойстиков
    user_wheel = start_work_wheel_pygame()
    virtual_wheel = start_work_virtual_gamepad()

    # Создаём камеру для скриншотов
    camera = dxcam.create(device_idx=0, output_idx=0, output_color='BGR')
    camera.start(target_fps=60)

    # Получаем области для распознавания
    road_area = get_road_area_size()
    speed_area = get_speed_area_size()

    # Объявляем нейросети и веса
    model_road, model_speed, model_wheel = get_weight_model()

    # Объявляем кнопки для управления
    first_key, second_key, third_key, fourth_key = get_key_for_management()

    combined_mask_old = torch.zeros((1, 1, 96, 128), device='cuda')
    combined_mask_new = torch.zeros((1, 1, 96, 128), device='cuda')

    array_wheel_pos = [i / 120 for i in range(0, 120, 2)]

    last_wheel_position = [0.0]

    opened = False
    recording = False

    queue_speed_pos = Queue()

    update_position = Thread(target=update_wheel_position_from_queue,
                             args=(virtual_wheel, queue_speed_pos, last_wheel_position),
                             daemon=True)
    update_position.start()

    stop_event = Event()

    update_process = Thread(target=update_virtual_wheel, args=(user_wheel, virtual_wheel, stop_event))

    update_process.start()

    print('Можно начинать!!!\n')

    try:
        while True:
            if recording:
                pygame.event.pump()
                array_wheel_position = [user_wheel.get_axis(0)] * 65

                queue_speed_pos.put(array_wheel_position)

                combined_mask_old.zero_()
                combined_mask_new.zero_()

                mass_all = []

                while opened:
                    start_time = time.time()

                    road_image, speed_image = save_photo(camera, road_area, speed_area)

                    combined_tensor, combined_mask_old, combined_mask_new = model_road_processing(
                        model_road, road_image, combined_mask_old, combined_mask_new)

                    speed = model_speed_processing(model_speed, speed_image)

                    # combined_tensor = torch.cat((combined_tensor, speed), dim=0)

                    new_tensor_wheel_position = model_wheel.forward(combined_tensor, speed)

                    array_wheel_position = interpolation_wheel_position(array_wheel_position, new_tensor_wheel_position,
                                                                        array_wheel_pos)
                    clear_queue(queue_speed_pos)

                    queue_speed_pos.put(array_wheel_position)

                    mass_all.append(time.time() - start_time)

                    if keyboard.is_pressed(f'{first_key}'):
                        recording = False
                        opened = False

                        print(sum(mass_all) / len(mass_all))

                        time.sleep(0.1)

                        print('Stop')

                        update_process = Thread(target=update_virtual_wheel, args=(user_wheel,
                                                                                   virtual_wheel,
                                                                                   stop_event))
                        stop_event.clear()

                        update_process.start()

            if keyboard.is_pressed(f'{first_key}'):
                recording = True
                opened = True

                stop_event.set()  # Устанавливаем флаг, сигнализирующий о необходимости остановки потока

                print('\nStart')

            if keyboard.is_pressed(f'{third_key}'):
                print('\nCalibration')

                x = [0.0, 0.2, 0.2, 0.4, 0.6, 0.9, 0.4, 0.3, 0.2, 0.0]
                for xi in x:
                    virtual_wheel.left_joystick_float(x_value_float=xi, y_value_float=0.0)
                    virtual_wheel.update()
                    time.sleep(0.01)

            if keyboard.is_pressed(f'{second_key}'):
                print('\nExit')
                clear_queue(queue_speed_pos)
                stop_event.clear()
                stop_event.set()
                break

    except KeyboardInterrupt:
        pygame.quit()
        cv2.destroyAllWindows()

    pygame.quit()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_code()
