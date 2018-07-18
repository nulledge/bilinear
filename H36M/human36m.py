import itertools
import math
import os
import pickle
import random

import cv2
import numpy as np
import torch
import torch.utils.data as torch_data
from PIL import Image
from torchvision import transforms
from vectormath import Vector2

from H36M.util import decode_image_name, rand, gaussian_3d
from .annotation import annotations, Annotation
from .task import tasks, Task

T = transforms.Compose([
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    transforms.ToTensor()
])


class Human36m(torch_data.Dataset):
    def __init__(self, image_path, subjects, task, joints,
                 heatmap_xy_coefficient, voxel_xy_resolution, voxel_z_resolutions, augment):
        self.voxel_z_res_list = torch.FloatTensor(voxel_z_resolutions)
        self.voxel_xy_res = voxel_xy_resolution
        self.heatmap_xy_coeff = heatmap_xy_coefficient
        self.task = task
        self.subjects = subjects
        self.joints = joints
        self.image_path = image_path
        self.augment = augment

        # initialize
        self.heatmap_z_coeff = 2 * torch.floor(
            (6 * self.heatmap_xy_coeff * self.voxel_z_res_list / self.voxel_z_res_list[-1] + 1) / 2) + 1

        self.gaussians = list()
        for z_coeff in self.heatmap_z_coeff:
            gau = gaussian_3d(3 * 2 * self.heatmap_xy_coeff + 1, z_coeff)
            self.gaussians.append(gau)

        self.voxels = list()
        for z_res in self.voxel_z_res_list:
            vx = torch.zeros(self.joints, z_res, self.voxel_xy_res, self.voxel_xy_res)
            self.voxels.append(vx)

        # read dataset appendix
        self.data = dict()
        for task in tasks:
            self.data[str(task)] = pickle.load(open("./%s.bin" % task, 'rb'))

    def __len__(self):
        return len(self.data[str(self.task)][str(Annotation.Image)])

    def __getitem__(self, index):
        raw_data = dict()
        for annotation in [Annotation.Image] + annotations[str(self.task)]:
            raw_data[str(annotation)] = self.data[str(self.task)][str(annotation)][index]
            if annotation is Annotation.Center: # and self.task is str(Task.Valid)
                raw_data[str(annotation)] = np.asarray([raw_data[str(annotation)].x, raw_data[str(annotation)].y])

        image, voxels, camera, sequence = self.preprocess(raw_data)

        if self.augment:
            for channel in range(3):
                image[channel, :, :] *= random.uniform(0.6, 1.4)
            image = np.clip(image, 0.0, 1.0)

        return image, voxels, camera, raw_data, sequence

    def __add__(self, item):
        pass

    def preprocess(self, raw_data):
        # Common annotations for training and validation.
        image_name = raw_data[str(Annotation.Image)]
        center = raw_data[str(Annotation.Center)]
        scale = raw_data[str(Annotation.Scale)]
        angle = 0

        # Data augmentation.
        if self.task == str(Task.Train) and self.augment:
            scale = scale * 2 ** rand(0.25) * 1.25
            angle = rand(30) if random.random() <= 0.4 else 0

        image_xy_res = 200 * scale

        # Extract subject and camera name from an image name.
        subject, sequence, camera, _ = decode_image_name(image_name)

        # Crop RGB image.
        image_path = os.path.join(self.image_path, subject, image_name)
        image = self._get_crop_image(image_path, center, scale, angle)

        if self.task == str(Task.Train):
            zind = np.clip(raw_data[str(Annotation.Z)], 1, 64)
            part = raw_data[str(Annotation.Part)]
            voxels = self._get_voxels(part, center, image_xy_res, angle, zind)
        else:
            voxels = -1

        return T(image), voxels, camera, sequence

    def _get_crop_image(self, image_path, center, scale, angle, resolution=256):
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

        if angle != 0:
            ul -= pad_length
            br += pad_length

        crop_src = [max(0, ul.x), max(0, ul.y), min(width, br.x), min(height, br.y)]
        crop_dst = [max(0, -ul.x), max(0, -ul.y), min(width, br.x) - ul.x, min(height, br.y) - ul.y]
        crop_image = image.crop(crop_src)

        new_image = Image.new("RGB", (br.x - ul.x, br.y - ul.y))
        new_image.paste(crop_image, box=crop_dst)

        if angle != 0:
            new_image = new_image.rotate(angle, resample=Image.BILINEAR)
            new_image = new_image.crop(box=(pad_length, pad_length,
                                            new_image.width - pad_length, new_image.height - pad_length))

        if crop_ratio < 2:
            new_image = new_image.resize((resolution, resolution), Image.BILINEAR)

        return new_image

    def _get_voxels(self, part, center, image_xy_res, angle, zidx):
        part = torch.from_numpy(part).float()
        center = torch.from_numpy(center).float()
        zidx = torch.from_numpy(zidx).float()

        # for vx in self.voxels:
        #     vx.zero_()
        voxels = list()
        for z_res in self.voxel_z_res_list:
            vx = torch.zeros(self.joints, z_res, self.voxel_xy_res, self.voxel_xy_res)
            voxels.append(vx)

        xy = self.voxel_xy_res * (part - center) / image_xy_res + self.voxel_xy_res * 0.5

        if angle != 0.0:
            xy = xy - self.voxel_xy_res / 2
            cos = math.cos(angle * math.pi / 180)
            sin = math.sin(angle * math.pi / 180)
            x = sin * xy[:, 1] + cos * xy[:, 0]
            y = cos * xy[:, 1] - sin * xy[:, 0]
            xy[:, 0] = x
            xy[:, 1] = y
            xy = xy + self.voxel_xy_res / 2

        zidx = torch.ceil(zidx.unsqueeze(-1) * self.voxel_z_res_list / self.voxel_z_res_list[-1]) - 1
        zidx = zidx.short().t()
        zpad = torch.floor(self.heatmap_z_coeff / 2).short()

        xy = xy.short()
        pad = 3 * self.heatmap_xy_coeff

        dst = [torch.clamp(xy - pad, min=0), torch.clamp(xy + pad + 1, max=self.voxel_xy_res, min=0)]
        src = [torch.clamp(pad - xy, min=0), pad + 1 + torch.clamp(self.voxel_xy_res - xy - 1, max=pad)]

        # z_res_cumsum = np.insert(np.cumsum(self.voxel_z_res_list), 0, 0)
        # voxels = np.zeros((len(part), z_res_cumsum[-1], self.voxel_xy_res, self.voxel_xy_res), dtype=np.float32)  # JVHW

        zdsts, zsrcs = list(), list()
        for z, z_res, pad in zip(zidx, self.voxel_z_res_list.short(), zpad):
            zdst = [torch.clamp(z - pad, min=0), torch.clamp(z + pad + 1, max=z_res, min=0)]  # BJ
            zsrc = [torch.clamp(pad - z, min=0), pad + 1 + torch.clamp(z_res - z - 1, max=pad)]  # BJ
            zdsts.append(zdst)
            zsrcs.append(zsrc)

        for (vx, zdst, zsrc, g), j in itertools.product(zip(voxels, zdsts, zsrcs, self.gaussians),
                                                        range(self.joints)):
            if xy[j, 0] < 0 or self.voxel_xy_res <= xy[j, 0] or \
                    xy[j, 1] < 0 or self.voxel_xy_res <= xy[j, 1]:
                continue
            z_dst_slice = slice(zdst[0][j], zdst[1][j])
            y_dst_slice = slice(dst[0][j, 1], dst[1][j, 1])
            x_dst_slice = slice(dst[0][j, 0], dst[1][j, 0])

            z_src_slice = slice(zsrc[0][j], zsrc[1][j])
            y_src_slice = slice(src[0][j, 1], src[1][j, 1])
            x_src_slice = slice(src[0][j, 0], src[1][j, 0])

            vx[j, z_dst_slice, y_dst_slice, x_dst_slice] = g[z_src_slice, y_src_slice, x_src_slice]

        return voxels
