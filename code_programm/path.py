import os
from datetime import datetime
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent


def get_path_save_screens(name_path, name_screen):
    return os.path.join(get_project_root(), f'D:\\Dataset_for_autopilot\\{name_path}\\{name_screen}')

def get_path_save_screens_road(name_path, name_screen):
    if not os.path.exists(fr'D:\\Dataset_for_autopilot\\{name_path}\\road'):
        os.makedirs(fr'D:\\Dataset_for_autopilot\\{name_path}\\road')
    return f'D:\\Dataset_for_autopilot\\{name_path}\\road\\{name_screen}'


def get_path_save_screens_speed(name_path, name_screen):
    if not os.path.exists(fr'D:\\Dataset_for_autopilot\\{name_path}\\speed'):
        os.makedirs(fr'D:\\Dataset_for_autopilot\\{name_path}\\speed')
    return f'D:\\Dataset_for_autopilot\\{name_path}\\speed\\{name_screen}'

def get_path_save_csv(path_name):
    return os.path.join(path_name, f'log.csv')

def get_path_to_dataset_for_yolo_line():
    return os.path.join(get_project_root(), f'static\\datasets_line\\data.yaml')

def get_path_to_dataset_for_yolo_numbers():
    return os.path.join(get_project_root(), f'static\\datasets_numbers\\data.yaml')


def get_path_config_road_area_size():
    return os.path.join(get_project_root(), f'static\\config\\config_road_area_size.txt')

def get_path_config_speed_area_size():
    return os.path.join(get_project_root(), f'static\\config\\config_road_area_size.txt')

def get_path_config_speed_area_size():
    return os.path.join(get_project_root(), f'static\\config\\config_speed_area_size.txt')

def get_path_name():
    folder = os.path.join(get_project_root(), f'D:\\Dataset_for_autopilot')
    name_new_folder = (f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}').replace(':', "-")
    name_new_path = os.path.join(folder, name_new_folder)
    if not os.path.exists(name_new_path):
        os.makedirs(name_new_path)
    return name_new_path, name_new_folder


def get_path_model(model_name):
    return os.path.join(get_project_root(), f'static\\models_yolo\\{model_name}')


def get_path_weight_model(weight_model_name):
    return os.path.join(get_project_root(), f'static\\weight_model\\{weight_model_name}')


def get_path_wheel_net_weight(dataset_model_name):
    return os.path.join(get_project_root(), f'dataset_for_wheel_nn\\{dataset_model_name}')
