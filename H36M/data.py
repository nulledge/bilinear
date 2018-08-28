import pickle
import math
import numpy as np
import torch.utils.data as torch_data
import os
from torchvision import transforms
from vectormath import Vector2

from .util import decode_image_name
from .annotation import annotations, Annotation
from .task import tasks, Task
from .util import draw_heatmap, crop_image


class Dataset(torch_data.Dataset):

    def __init__(self, data_dir, task, position_only=True):

        assert task in tasks
        assert os.path.exists(data_dir) and 'Human3.6M' in data_dir

        self.data_dir = data_dir
        self.task = task
        self.position_only = position_only

        self.data, self.mean, self.stddev = (dict(), dict(), dict())
        for task in tasks:
            self.data[task] = pickle.load(open("{data_dir}/{task}.bin".format(data_dir=self.data_dir, task=task), 'rb'))
            self.mean[task] = dict()
            self.stddev[task] = dict()
            for dim in [2, 3]:
                self.mean[task][dim], self.stddev[task][dim] = self.normalize(task=task, dim=dim)

    def __len__(self):
        return len(self.data[self.task][Annotation.Image])

    def __getitem__(self, index):
        data = dict()
        for annotation in [Annotation.Image] + annotations[self.task]:
            data[annotation] = self.data[self.task][annotation][index]

            if annotation is Annotation.Center:  # Correct annotation.
                data[annotation] = np.asarray([data[annotation].x, data[annotation].y])

        if self.position_only:
            image, heatmap = [-1, -1]
        else:
            image, heatmap = self.preprocess(data)

        for dim, anno in zip([2, 3], [Annotation.Part, Annotation.S]):
            data[anno] = (data[anno] - self.mean[self.task][dim]) / self.stddev[self.task][dim]

        return data[Annotation.Part], data[Annotation.S], \
               data[Annotation.Center], data[Annotation.Scale], \
               image, heatmap

    def __add__(self, item):
        pass

    def preprocess(self, data):
        # Common annotations for training and validation.
        image_name = data[Annotation.Image]
        center = data[Annotation.Center]
        scale = data[Annotation.Scale]
        part = data[Annotation.Part]
        angle = 0

        # Extract subject from an image name.
        subject, _, _, _ = decode_image_name(image_name)

        # Crop RGB image.
        image_path = '{data_dir}/{subject}/{image_name}'.format(data_dir=self.data_dir, subject=subject, image_name=image_name)
        image = crop_image(image_path, center, scale, angle)

        if self.task == Task.Train:
            heatmap = np.zeros(shape=(17, 64, 64), dtype=np.float32)

            for idx, keypoint in enumerate(part):
                in_image = Vector2(keypoint[0], keypoint[1])
                in_heatmap = (in_image - center) * 64 / (200 * scale)

                if angle != 0:
                    cos = math.cos(angle * math.pi / 180)
                    sin = math.sin(angle * math.pi / 180)
                    in_heatmap = Vector2(sin * in_heatmap.y + cos * in_heatmap.x,
                                         cos * in_heatmap.y - sin * in_heatmap.x)

                in_heatmap = in_heatmap + Vector2(64 // 2, 64 // 2)

                if min(in_heatmap) < 0 or max(in_heatmap) >= 64:
                    continue

                heatmap[idx, :, :] = draw_heatmap(64, in_heatmap.y, in_heatmap.x)
        else:
            heatmap = -1

        return np.asarray(image), heatmap

    def normalize(self, task, dim):
        assert task in tasks
        assert dim in [2, 3]

        if dim == 3:
            anno = Annotation.S
        else:
            anno = Annotation.Part

        root = self.data[task][anno][:][0]  # Frame-Dim
        root_centered = self.data[task][anno] - root  # Frame-Joint-Dim
        self.data[task][anno] = root_centered

        data = np.reshape(np.asarray(self.data[task][anno]), newshape=(-1, dim * 17))  # Frame-Dim*Joint
        mean = np.reshape(np.mean(data, axis=0), newshape=(17, dim))  # Joint-Dim
        stddev = np.reshape(np.std(data, axis=0), newshape=(17, dim))  # Joint-Dim

        return mean, stddev
