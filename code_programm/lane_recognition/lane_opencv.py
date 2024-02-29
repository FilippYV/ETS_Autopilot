import time

import cv2


def canny(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.Canny(blur, 50, 200)


if __name__ == '__main__':
    start_time = time.time()
    img = cv2.imread('../../static/settings_cache/17.png')
    img_copy = img.copy()
    img = canny(img)
    print('stop_time:', time.time()-start_time)
    cv2.imshow('photo', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
