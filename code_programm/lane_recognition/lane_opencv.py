import time

import cv2

if __name__ == '__main__':
    start_time = time.time()
    img = cv2.imread(r'C:\PycharmProjects\ETS_Autopilot\static\settings_cache\2024-04-02 19-12-38_0.png')
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)
    print('stop_time:', time.time() - start_time)
    cv2.imshow('photo', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
