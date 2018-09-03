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

        self.data = dict()
        for task in tasks:
            self.data[task] = pickle.load(open("{data_dir}/{task}.bin".format(data_dir=self.data_dir, task=task), 'rb'))

            for anno in [Annotation.Part, Annotation.S]:
                self.data[task][anno], \
                self.data[task][Annotation.Root_Of + anno], \
                self.data[task][Annotation.Mean_Of + anno], \
                self.data[task][Annotation.Stddev_Of + anno] = self.normalize(task=task, anno=anno)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data[self.task][Annotation.Image])

    def __getitem__(self, index):
        data = dict()
        required = [
                       Annotation.Image,
                       Annotation.Root_Of + Annotation.Part,
                       Annotation.Mean_Of + Annotation.Part,
                       Annotation.Stddev_Of + Annotation.Part,
                   ] + annotations[self.task]

        for annotation in required:
            if Annotation.Mean_Of in annotation or Annotation.Stddev_Of in annotation:
                select = slice(None)
            else:
                select = index

            data[annotation] = self.data[self.task][annotation][select]

            if annotation == Annotation.Center:  # Correct annotation.
                data[annotation] = np.asarray([data[annotation].x, data[annotation].y], dtype=np.float32)
            if annotation == Annotation.Scale:
                data[annotation] = np.float32(data[annotation])

        if self.position_only:
            image, heatmap = -1, -1
        else:
            image, heatmap = self.preprocess(data)

        for dim, anno in zip([2, 3], [Annotation.Part, Annotation.S]):
            data[anno] = data[anno] - self.data[self.task][Annotation.Mean_Of + anno]
            data[anno] = data[anno] / self.data[self.task][Annotation.Stddev_Of + anno]

        return data[Annotation.Part], data[Annotation.S], \
               data[Annotation.Center], data[Annotation.Scale], \
               data[Annotation.Root_Of + Annotation.Part], \
               data[Annotation.Mean_Of + Annotation.Part], \
               data[Annotation.Stddev_Of + Annotation.Part], \
               image, heatmap

    def __add__(self, item):
        pass

    def preprocess(self, data):
        # Common annotations for training and validation.
        image_name = data[Annotation.Image]
        center = Vector2(data[Annotation.Center])
        scale = data[Annotation.Scale]
        # Concat [[0, 0]] for pelvis
        part = np.concatenate([np.zeros(shape=(1, 2)), data[Annotation.Part]])
        # Root is shape of (1, 2)
        root = data[Annotation.Root_Of + Annotation.Part].squeeze()
        angle = 0

        # Extract subject from an image name.
        subject, _, _, _ = decode_image_name(image_name)

        # Crop RGB image.
        image_path = '{data_dir}/{subject}/{image_name}'.format(data_dir=self.data_dir, subject=subject,
                                                                image_name=image_name)
        image = crop_image(image_path, center, scale, angle)

        if self.task == Task.Train:
            heatmap = np.zeros(shape=(17, 64, 64), dtype=np.float32)

            for idx, keypoint in enumerate(part):
                # Un-normalize
                in_image = Vector2(keypoint) + Vector2(root)
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

        return self.transform(image), heatmap

    def normalize(self, task, anno):
        assert task in tasks
        assert anno in [Annotation.Part, Annotation.S]

        if anno == Annotation.S:
            dim = 3
        else:
            dim = 2

        data = np.asarray(self.data[task][anno], dtype=np.float32)

        root = data[:, 0, :]  # Frame-Dim
        root = np.expand_dims(root, 1)  # Frame-Joint-Dim
        root_centered = data - root  # Frame-Joint-Dim
        root_removed = root_centered[:, 1:, :]  # Frame-Joint(pelvis removed)-Dim

        assert root.shape[1:] == (1, dim)
        assert root_removed.shape[1:] == (17 - 1, dim)

        data = np.reshape(root_removed, newshape=(-1, dim * (17 - 1)))  # Frame-Dim*Joint
        mean = np.reshape(np.mean(data, axis=0), newshape=(17 - 1, dim))  # Joint-Dim
        stddev = np.reshape(np.std(data, axis=0), newshape=(17 - 1, dim))  # Joint-Dim

        return root_removed, root, mean, stddev
