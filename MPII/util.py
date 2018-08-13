import math
import numpy as np
import skimage, skimage.transform
from functools import lru_cache
from PIL import Image
from random import gauss
from vectormath import Vector2


def rand(x):
    return max(-2 * x, min(2 * x, gauss(0, 1) * x))


def crop_image(image, center, scale, rotate, resolution):
    center = Vector2(center)  # assign new array
    height, width, _ = image.shape
    crop_ratio = 200 * scale / resolution
    if crop_ratio >= 2:  # if box size is greater than two time of resolution px
        # scale down image
        height = math.floor(height / crop_ratio)
        width = math.floor(width / crop_ratio)

        if max([height, width]) < 2:
            # Zoomed out so much that the image is now a single pixel or less
            raise ValueError("Width or height is invalid!")

        image = skimage.transform.resize(image, (height, width), mode='constant')
        center /= crop_ratio
        scale /= crop_ratio

    ul = (center - 200 * scale / 2).astype(int)
    br = (center + 200 * scale / 2).astype(int)  # Vector2

    if crop_ratio >= 2:  # force image size 256 x 256
        br -= (br - ul - resolution)

    pad_length = math.ceil((ul - br).length - (br.x - ul.x) / 2)

    if rotate != 0:
        ul -= pad_length
        br += pad_length

    src = [max(0, ul.y), min(height, br.y), max(0, ul.x), min(width, br.x)]
    dst = [max(0, -ul.y), min(height, br.y) - ul.y, max(0, -ul.x), min(width, br.x) - ul.x]

    new_image = np.zeros([br.y - ul.y, br.x - ul.x, 3], dtype=np.float64)
    new_image[dst[0]:dst[1], dst[2]:dst[3], :] = image[src[0]:src[1], src[2]:src[3], :]

    if rotate != 0:
        new_image = skimage.transform.rotate(new_image, rotate)
        new_height, new_width, _ = new_image.shape
        new_image = new_image[pad_length:new_height - pad_length, pad_length:new_width - pad_length, :]

    if crop_ratio < 2:
        new_image = skimage.transform.resize(new_image, (resolution, resolution), mode='constant')

    return new_image


@lru_cache(maxsize=32)
def gaussian(size, sigma=0.25, mean=0.5):
    width = size
    height = size
    amplitude = 1.0
    sigma_u = sigma
    sigma_v = sigma
    mean_u = mean * width + 0.5
    mean_v = mean * height + 0.5

    over_sigma_u = 1.0 / (sigma_u * width)
    over_sigma_v = 1.0 / (sigma_v * height)

    x = np.arange(0, width, 1)
    y = x[:, np.newaxis]

    du = (x + 1 - mean_u) * over_sigma_u
    dv = (y + 1 - mean_v) * over_sigma_v

    return amplitude * np.exp(-0.5 * (du * du + dv * dv))


def draw_heatmap(size, y0, x0, sigma=1):
    pad = 3 * sigma
    y0, x0 = int(y0), int(x0)
    dst = [max(0, y0 - pad), max(0, min(size, y0 + pad + 1)), max(0, x0 - pad), max(0, min(size, x0 + pad + 1))]
    src = [-min(0, y0 - pad), pad + min(pad, size - y0 - 1) + 1, -min(0, x0 - pad), pad + min(pad, size - x0 - 1) + 1]

    heatmap = np.zeros([size, size])
    g = gaussian(3 * 2 * sigma + 1)
    heatmap[dst[0]:dst[1], dst[2]:dst[3]] = g[src[0]:src[1], src[2]:src[3]]

    return heatmap
