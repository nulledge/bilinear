import pickle
import math
import numpy as np
import torch.utils.data as torch_data
import os
from torchvision import transforms
from vectormath import Vector2

from .util import decode_image_name
from .annotation import Annotation
from .protocol import Protocol
from .task import tasks, Task
from .util import draw_heatmap, crop_image


class Dataset(torch_data.Dataset):

    def __init__(self, data_dir, task, position_only=True, protocol=Protocol.GT):

        assert task in tasks
        assert protocol in [Protocol.GT, Protocol.SH, Protocol.SH_FT]
        assert os.path.exists(data_dir) and 'Human3.6M' in data_dir

        self.data_dir = data_dir
        self.task = task
        self.position_only = position_only
        self.protocol = protocol

        self.data = dict()
        for task in [Task.Train, Task.Valid]:

            data_path = "{data_dir}/{task}_{protocol}.bin".format(data_dir=data_dir, task=task, protocol=protocol)
            self.data[task] = pickle.load(open(data_path, 'rb'))

            for dim, anno in zip([2, 3], [Annotation.Part, Annotation.S]):

                # self.data['part'] and self.data['S'] are list object
                self.data[task][anno] = np.asarray(self.data[task][anno], dtype=np.float32)

                if anno == Annotation.Part:
                    # We only use 16 joint, except for nose
                    self.data[task][anno] = np.delete(self.data[task][anno], 9, axis=1)  # Shape=(n_data, 16, 2)

                # Root-center normalization for 3D pose
                elif anno == Annotation.S:
                    root = self.data[task][anno][:, 0, :]  # Frame-Dim
                    root = np.expand_dims(root, 1)  # Frame-Joint-Dim
                    root_centered = self.data[task][anno] - root
                    self.data[task][anno] = root_centered  # Shape=(n_data, 17, 2)

                    # Remove the pelvis because mean and stddev after root-centering are always 0.0
                    root_removed = self.data[task][anno][:, 1:, :]
                    self.data[task][anno] = root_removed  # Shape=(n_data, 16, 2)

                # Calculate mean and stddev
                self.data[task][anno] = np.reshape(self.data[task][anno], newshape=(-1, dim * 16))  # Frame-Dim*Joint
                self.data[task][Annotation.Mean_Of + anno] = np.mean(self.data[task][anno], axis=0)  # Dim*Joint
                self.data[task][Annotation.Stddev_Of + anno] = np.std(self.data[task][anno], axis=0)  # Dim*Joint

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data[self.task][Annotation.Image])

    def __getitem__(self, index):
        data = dict()
        required = [
            Annotation.Image,
            Annotation.S,
            Annotation.Center,
            Annotation.Part,
            Annotation.Scale,
            Annotation.Mean_Of + Annotation.S,
            Annotation.Stddev_Of + Annotation.S,
        ]

        for annotation in required:
            if Annotation.Mean_Of in annotation or Annotation.Stddev_Of in annotation:
                select = slice(None)
                task = Task.Train
            else:
                select = index
                task = self.task

            data[annotation] = self.data[task][annotation][select]

            if annotation == Annotation.Center:  # Correct annotation.
                data[annotation] = np.asarray([data[annotation].x, data[annotation].y], dtype=np.float32)
            if annotation == Annotation.Scale:
                data[annotation] = np.float32(data[annotation])

        if self.position_only:
            image, heatmap = -1, -1
        else:
            image, heatmap = self.preprocess(data)

        _, action, _, _ = decode_image_name(data[Annotation.Image])

        for dim, anno in zip([2, 3], [Annotation.Part, Annotation.S]):
            data[anno] = data[anno] - self.data[Task.Train][Annotation.Mean_Of + anno]
            data[anno] = data[anno] / self.data[Task.Train][Annotation.Stddev_Of + anno]

        return data, image, heatmap, action

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
            heatmap = np.zeros(shape=(16, 64, 64), dtype=np.float32)

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
