import cv2
import torch
from ultralytics import YOLO

from code_programm.path import get_path_weight_model

print(torch.cuda.device_count())
print(torch.cuda.get_device_name())

model = YOLO(get_path_weight_model('best_v2_n.pt'))

img = cv2.imread(r'C:\PycharmProjects\ETS_Autopilot\static\settings_cache\1708818112.041754.png')
print(img.shape)

results = model(img, imgsz=512, conf=0.75, verbose=True, show=True, device='cpu')

H, W, _ = img.shape

for result in results:
    for j, mask in enumerate(result.masks.data):
        mask = mask.numpy() * 255
        mask = cv2.resize(mask, (W, H))
        cv2.imwrite('output.png', mask)
