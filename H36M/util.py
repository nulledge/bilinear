import math
import random
from functools import lru_cache

import numpy as np
import torch
from vectormath import Vector2


@lru_cache(maxsize=32)
def gaussian_3d(size, depth, sigma=0.25, mean=0.5, amplitude=1.0, device='cpu'):
    width = size
    height = size

    sigma_u = sigma
    sigma_v = sigma
    sigma_r = sigma

    mean_u = mean * width + 0.5
    mean_v = mean * height + 0.5
    mean_r = mean * depth + 0.5

    over_sigma_u = 1.0 / (sigma_u * width)
    over_sigma_v = 1.0 / (sigma_v * height)
    over_sigma_r = 1.0 / (sigma_r * depth)

    x = torch.arange(0, width, 1, dtype=torch.float32).to(device)
    y = x.view(-1, 1)
    z = torch.arange(0, depth, 1, dtype=torch.float32).to(device)
    z = z.view(-1, 1, 1)

    du = (x + 1 - mean_u) * over_sigma_u
    dv = (y + 1 - mean_v) * over_sigma_v
    dr = (z + 1 - mean_r) * over_sigma_r

    gau = amplitude * torch.exp(-0.5 * (du * du + dv * dv + dr * dr))

    return gau  # .transpose(1, 2, 0)


@lru_cache(maxsize=32)
def gaussian_3d_deprecated(size, depth, sigma=0.25, mean=0.5, amplitude=1.0):
    width = size
    height = size

    sigma_u = sigma
    sigma_v = sigma
    sigma_r = sigma

    mean_u = mean * width + 0.5
    mean_v = mean * height + 0.5
    mean_r = mean * depth + 0.5

    over_sigma_u = 1.0 / (sigma_u * width)
    over_sigma_v = 1.0 / (sigma_v * height)
    over_sigma_r = 1.0 / (sigma_r * depth)

    x = np.arange(0, width, 1, dtype=np.float32)
    y = x[:, np.newaxis]
    z = np.arange(0, depth, 1, dtype=np.float32)
    z = z[:, np.newaxis, np.newaxis]

    du = (x + 1 - mean_u) * over_sigma_u
    dv = (y + 1 - mean_v) * over_sigma_v
    dr = (z + 1 - mean_r) * over_sigma_r

    gau = amplitude * np.exp(-0.5 * (du * du + dv * dv + dr * dr))

    return gau  # .transpose(1, 2, 0)


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

    x = np.arange(0, width, 1, dtype=np.float32)
    y = x[:, np.newaxis]

    du = (x + 1 - mean_u) * over_sigma_u
    dv = (y + 1 - mean_v) * over_sigma_v

    return amplitude * np.exp(-0.5 * (du * du + dv * dv))


def set_heatmap_3d(heatmap, size, y0, x0, sigma=1):
    pad = 3 * sigma
    y0, x0 = int(y0), int(x0)
    dst = [max(0, y0 - pad), max(0, min(size, y0 + pad + 1)), max(0, x0 - pad), max(0, min(size, x0 + pad + 1))]
    src = [-min(0, y0 - pad), pad + min(pad, size - y0 - 1) + 1, -min(0, x0 - pad), pad + min(pad, size - x0 - 1) + 1]

    # heatmap = np.zeros([size, size], dtype=np.float32)
    g = gaussian(3 * 2 * sigma + 1)
    heatmap[dst[0]:dst[1], dst[2]:dst[3]] = g[src[0]:src[1], src[2]:src[3]]

    # return heatmap


def generate_heatmap(size, y0, x0, sigma=1):
    pad = 3 * sigma
    y0, x0 = int(y0), int(x0)
    dst = [max(0, y0 - pad), max(0, min(size, y0 + pad + 1)), max(0, x0 - pad), max(0, min(size, x0 + pad + 1))]
    src = [-min(0, y0 - pad), pad + min(pad, size - y0 - 1) + 1, -min(0, x0 - pad), pad + min(pad, size - x0 - 1) + 1]

    heatmap = np.zeros([size, size], dtype=np.float32)
    g = gaussian(3 * 2 * sigma + 1)
    heatmap[dst[0]:dst[1], dst[2]:dst[3]] = g[src[0]:src[1], src[2]:src[3]]

    return heatmap


def crop_image(image, center, scale, rotate, resolution):
    center = Vector2(center)  # assign new array
    height, width, channel = image.shape
    crop_ratio = 200 * scale / resolution
    if crop_ratio >= 2:  # if box size is greater than two time of resolution px
        # scale down image
        height = math.floor(height / crop_ratio)
        width = math.floor(width / crop_ratio)

        if max([height, width]) < 2:
            # Zoomed out so much that the image is now a single pixel or less
            raise ValueError("Width or height is invalid!")

        # image = skimage.transform.resize(image, (height, width), mode='constant')
        image = image.resize(image, (height, width), mode='constant')
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

    new_image = np.zeros([br.y - ul.y, br.x - ul.x, channel], dtype=np.float32)
    new_image[dst[0]:dst[1], dst[2]:dst[3], :] = image[src[0]:src[1], src[2]:src[3], :]

    if rotate != 0:
        new_image = skimage.transform.rotate(new_image, rotate)
        new_height, new_width, _ = new_image.shape
        new_image = new_image[pad_length:new_height - pad_length, pad_length:new_width - pad_length, :]

    if crop_ratio < 2:
        # new_image = skimage.transform.resize(new_image, (resolution, resolution), mode='constant')
        new_image = image.resize(new_image, (resolution, resolution), mode='constant')

    return new_image


def set_voxel(volume, voxel_xy_res, voxel_z_res, xy, z, heatmap_xy_coeff, heatmap_z_coeff):
    pad = 3 * heatmap_xy_coeff
    zpad = math.floor(heatmap_z_coeff / 2)
    y0, x0 = int(xy[1]), int(xy[0])
    dst = [max(0, y0 - pad), max(0, min(voxel_xy_res, y0 + pad + 1)),
           max(0, x0 - pad), max(0, min(voxel_xy_res, x0 + pad + 1)),
           max(0, z - zpad), max(0, min(voxel_z_res, z + zpad + 1))]
    src = [max(0, pad - y0), pad + min(pad, voxel_xy_res - y0 - 1) + 1,
           max(0, pad - x0), pad + min(pad, voxel_xy_res - x0 - 1) + 1,
           max(0, zpad - z), zpad + min(zpad, voxel_z_res - z - 1) + 1]

    g = gaussian_3d(3 * 2 * heatmap_xy_coeff + 1, heatmap_z_coeff)
    volume[dst[4]:dst[5], dst[0]:dst[1], dst[2]:dst[3]] = g[src[4]:src[5], src[0]:src[1], src[2]:src[3]]
    # print("dst z: ", dst[4],dst[5], "\tsrc z:", src[4],src[5])
    # volume = np.zeros(shape=(voxel_xy_res, voxel_xy_res, voxel_z_res), dtype=np.float32)
    # set_heatmap_3d(size=voxel_xy_res, y0=xy[1], x0=xy[0], sigma=heatmap_xy_coeff)
    # z_view = gaussian(heatmap_z_coeff)[math.ceil(heatmap_z_coeff / 2) - 1]
    # cnt = 0
    # for i in range(z - math.floor(heatmap_z_coeff / 2), z + math.floor(heatmap_z_coeff / 2) + 1):
    #     if 0 <= i < voxel_z_res:
    #         volume[:, :, i] = z_view[cnt] * xy_view
    #     cnt = cnt + 1
    # return volume


def generate_voxel(voxel_xy_res, voxel_z_res, xy, z, heatmap_xy_coeff, heatmap_z_coeff):
    volume = np.zeros(shape=(voxel_xy_res, voxel_xy_res, voxel_z_res), dtype=np.float32)
    xy_view = generate_heatmap(size=voxel_xy_res, y0=xy[1], x0=xy[0], sigma=heatmap_xy_coeff)
    z_view = gaussian(heatmap_z_coeff)[math.ceil(heatmap_z_coeff / 2) - 1]
    cnt = 0
    for i in range(z - math.floor(heatmap_z_coeff / 2), z + math.floor(heatmap_z_coeff / 2) + 1):
        if 0 <= i < voxel_z_res:
            volume[:, :, i] = z_view[cnt] * xy_view
        cnt = cnt + 1
    return volume


def decode_image_name(image_name):
    subject_action, camera_frame, _ = image_name.split('.')
    split = subject_action.split('_')
    subject = split[0]
    action = split[1]
    if len(split) >= 3:
        action = action + '_' + split[2]
    camera, frame = camera_frame.split('_')

    return subject, action, camera, frame


def rand(x):
    return max(-2 * x, min(2 * x, random.gauss(0, 1) * x))
