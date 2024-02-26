import os
from datetime import datetime
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent


def get_path_save_screens(name_path, name_screen):
    return os.path.join(get_project_root(), f'app\\static\\raw_data\\{name_path}\\{name_screen}')


def get_path_save_csv(path_name):
    return os.path.join(path_name, f'log.csv')


def get_path_to_dataset_for_yolo_train():
    return os.path.join('E:\\Dataset_for_Autopilot\\Version_0\\Marked', f'train')


def get_path_to_dataset_for_yolo_valid():
    return os.path.join('E:\\Dataset_for_Autopilot\\Version_0\\Marked', f'valid')


def get_path_to_dataset_for_yolo():
    return os.path.join(get_project_root(), f'app\\static\\datasets\\data.yaml')


def get_path_name():
    folder = os.path.join(get_project_root(), f'app\\static\\raw_data')
    name_new_folder = (f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}').replace(':', "-")
    name_new_path = os.path.join(folder, name_new_folder)
    if not os.path.exists(name_new_path):
        os.makedirs(name_new_path)
    return name_new_path, name_new_folder


def get_path_model(model_name):
    return os.path.join(get_project_root(), f'app\\model\\{model_name}')


def get_path_weight_model(weight_model_name):
    return os.path.join(get_project_root(), f'app\\weight_model\\{weight_model_name}')
