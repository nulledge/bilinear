import numpy as np


def centerCrop(img):
    imgArray = np.array(img)
    h, w, c = imgArray.shape
    size = w if h > w else h
    horizontal_center = w // 2
    vertical_center = h // 2
    anchor = [vertical_center - size // 2, horizontal_center - size // 2]
    retImgArray = imgArray[anchor[0]:anchor[0] + size, anchor[1]:anchor[1] + size, :]
    return retImgArray
