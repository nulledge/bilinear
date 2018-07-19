import math
import numpy as np
import skimage.transform
import torch
from functools import lru_cache
from PIL import Image
from random import gauss
from vectormath import Vector2
from visdom import Visdom

viz = Visdom()


def rand(x):
    return max(-2 * x, min(2 * x, gauss(0, 1) * x))


def crop_image(image_path, center, scale, rotate, resolution=256):
    image = Image.open(image_path)

    width, height = image.size
    center = Vector2(center)  # assign new array

    # scale = scale * 1.25
    crop_ratio = 200 * scale / resolution

    if crop_ratio >= 2:  # if box size is greater than two time of resolution px
        # scale down image
        height = math.floor(height / crop_ratio)
        width = math.floor(width / crop_ratio)

        if max([height, width]) < 2:
            # Zoomed out so much that the image is now a single pixel or less
            raise ValueError("Width or height is invalid!")

        image = image.resize((width, height), Image.BILINEAR)

        center /= crop_ratio
        scale /= crop_ratio

    ul = (center - 200 * scale / 2).astype(int)
    br = (center + 200 * scale / 2).astype(int)  # Vector2

    if crop_ratio >= 2:  # force image size 256 x 256
        br -= (br - ul - resolution)

    pad_length = math.ceil(((ul - br).length - (br.x - ul.x)) / 2)

    if rotate != 0:
        ul -= pad_length
        br += pad_length

    crop_src = [max(0, ul.x), max(0, ul.y), min(width, br.x), min(height, br.y)]
    crop_dst = [max(0, -ul.x), max(0, -ul.y), min(width, br.x) - ul.x, min(height, br.y) - ul.y]
    crop_image = image.crop(crop_src)

    new_image = Image.new("RGB", (br.x - ul.x, br.y - ul.y))
    new_image.paste(crop_image, box=crop_dst)

    if rotate != 0:
        new_image = new_image.rotate(rotate, resample=Image.BILINEAR)
        new_image = new_image.crop(box=(pad_length, pad_length,
                                        new_image.width - pad_length, new_image.height - pad_length))

    if crop_ratio < 2:
        new_image = new_image.resize((resolution, resolution), Image.BILINEAR)

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


def merge_to_color_heatmap(batch_heatmaps, h_format='NCHW'):
    Color = torch.cuda.FloatTensor(
        [[0, 0, 0.5],
         [0, 0, 1],
         [0, 1, 0],
         [1, 1, 0],
         [1, 0, 0]]
    )

    if h_format == 'NHWC':
        batch_heatmaps = batch_heatmaps.permute(0, 3, 1, 2).contiguous()

    batch, joints, height, width = batch_heatmaps.size()

    heatmaps = batch_heatmaps.clamp(0, 1.).view(-1)

    frac = torch.div(heatmaps, 0.25)
    lower_indices, upper_indices = torch.floor(frac).long(), torch.ceil(frac).long()

    t = frac - torch.floor(frac)
    t = t.view(-1, 1)

    k = Color.index_select(0, lower_indices)
    k_1 = Color.index_select(0, upper_indices)

    color_heatmap = (1.0 - t) * k + t * k_1
    color_heatmap = color_heatmap.view(batch, joints, height, width, 3)
    color_heatmap = color_heatmap.permute(0, 4, 2, 3, 1)
    color_heatmap, _ = torch.max(color_heatmap, 4)

    return color_heatmap


def draw_line(x, y, window):
    assert viz.check_connection()

    return viz.line(X=x,
                    Y=y,
                    win=window,
                    update='append' if window is not None else None)


def draw_merged_image(heatmaps, images, window):
    assert viz.check_connection()

    heatmaps = merge_to_color_heatmap(heatmaps.data)
    heatmaps = heatmaps.permute(0, 2, 3, 1).cpu()  # NHWC

    resized_heatmaps = list()
    for idx, ht in enumerate(heatmaps):
        color_ht = skimage.transform.resize(ht.numpy(), (256, 256), mode='constant')
        resized_heatmaps.append(color_ht.transpose(2, 0, 1))

    resized_heatmaps = np.stack(resized_heatmaps, axis=0)

    images = images * 0.6
    overlayed_image = np.clip(images + resized_heatmaps * 0.4, 0, 1.)

    return viz.images(tensor=overlayed_image, nrow=4, win=window)
