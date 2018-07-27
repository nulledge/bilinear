import pickle
import math
import numpy as np
import torch.utils.data as torch_data
from torchvision import transforms
from vectormath import Vector2

from .util import decode_image_name
from .annotation import annotations, Annotation
from .task import tasks, Task
from .util import draw_heatmap, crop_image


class Dataset(torch_data.Dataset):
    to_tensor = transforms.ToTensor()

    def __init__(self, root, task, end2end=False):
        self.root = root
        self.task = task
        self.end2end = end2end

        self.data = dict()
        for task in tasks:
            self.data[task] = pickle.load(open("{root}/{task}.bin".format(root=root, task=task), 'rb'))

        self.mean = dict()
        self.stddev = dict()
        for dim in [2, 3]:
            self.mean[dim], self.stddev[dim] = self.normalize(dim=dim)

    def __len__(self):
        return len(self.data[self.task][Annotation.Image])

    def __getitem__(self, index):
        data = dict()
        for annotation in [Annotation.Image] + annotations[self.task]:
            data[annotation] = self.data[self.task][annotation][index]

            if annotation is Annotation.Center:  # Correct annotation.
                data[annotation] = np.asarray([data[annotation].x, data[annotation].y])

        if self.end2end:
            image, heatmap = self.preprocess(data)
        else:
            image, heatmap = [-1, -1]

        for dim, anno in zip([2, 3], [Annotation.Part, Annotation.S]):
            data[anno] = data[anno] - data[anno][0]  # root-centered
            data[anno] = (data[anno] - self.mean[dim]) / self.stddev[dim]  # normalize for each joint and coord.
            data[anno] = np.asarray(data[anno], dtype=np.float32)

        return (data[Annotation.Part], data[Annotation.S],
                data[Annotation.Center], data[Annotation.Scale],
                image, heatmap)

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
        image_path = '{root}/{subject}/{image_name}'.format(root=self.root, subject=subject, image_name=image_name)
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

    def normalize(self, dim):
        if dim == 3:
            anno = Annotation.S
        else:
            anno = Annotation.Part
        data = np.reshape(np.asarray(self.data[self.task][anno]), newshape=(-1, dim * 17))
        mean = np.reshape(np.mean(data, axis=0), newshape=(-1, dim))
        stddev = np.reshape(np.std(data, axis=0), newshape=(-1, dim))

        return mean, stddev
