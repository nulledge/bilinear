import math
import numpy as np
import os
import scipy, scipy.io
import torch.utils.data as torch_data
from random import random, shuffle
from torchvision import transforms
from vectormath import Vector2

from .util import rand, crop_image, draw_heatmap
from MPII.task import Task


class Dataset(torch_data.Dataset):

    def __init__(self, root, task, augment=True):
        self.root = root
        self.task = task
        self.augment = augment

        self.image_path = '{root}/images'.format(root=root)

        self.annotation_path = '{root}/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'.format(root=root)
        self.annotation = scipy.io.loadmat(self.annotation_path, squeeze_me=True, struct_as_record=False)
        self.annotation = self.annotation['RELEASE']

        self.subset_path = '{root}/MPII-{task}.txt'.format(root=root, task=task)
        if not os.path.exists(self.subset_path):
            self.refresh_subset()
        self.subset = np.loadtxt(self.subset_path, dtype=np.int32)

        self.transform = transforms.ToTensor()

        if self.task == Task.Train and self.augment:
            self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def refresh_subset(self):
        correct = list()
        annotated = self.annotation.img_train
        index = np.arange(len(annotated))
        for img_idx in index[annotated != 0]:
            assert annotated[img_idx] == 1

            annorect = self.annotation.annolist[img_idx].annorect

            if type(annorect) != np.ndarray:
                annorect = [annorect]

            for r_idx in range(len(annorect)):
                # noinspection PyBroadException
                try:
                    # Check if the annotation is correct.
                    assert annorect[r_idx].objpos.y

                    correct.append((img_idx, r_idx))

                except Exception:
                    continue

        shuffle(correct)
        correct = np.asarray(correct)
        n_train = int(0.9 * len(correct))

        train_subset_path = '{root}/MPII-train.txt'.format(root=self.root)
        valid_subset_path = '{root}/MPII-valid.txt'.format(root=self.root)

        np.savetxt(train_subset_path, correct[:n_train], fmt='%u')
        np.savetxt(valid_subset_path, correct[n_train:], fmt='%u')

    def __getitem__(self, index):
        img_idx, r_idx = self.subset[index]

        annolist = self.annotation.annolist
        annorect = annolist[img_idx].annorect

        # Unify the shape.
        if type(annorect) != np.ndarray:
            annorect = [annorect]

        annorect = annorect[r_idx]

        # Rotation and scaling augmentation factors.
        scale = 1.25 * annorect.scale
        rotate = 0.0
        if self.task == Task.Train and self.augment:
            scale = scale * 2 ** rand(0.25)
            rotate = rand(30) if random() <= 0.4 else 0.0

        objpos = annorect.objpos
        center = Vector2(objpos.x, objpos.y + 15 * annorect.scale)
        center.setflags(write=False)

        head = 0.6 * math.sqrt((annorect.x2 - annorect.x1) ** 2 + (annorect.y2 - annorect.y1) ** 2)

        image_name = annolist[img_idx].image.name
        image_path = '{image_path}/{image_name}'.format(image_path=self.image_path, image_name=image_name)
        image = crop_image(image_path, center, scale, rotate)

        position = np.zeros(shape=(16, 2), dtype=np.float32)
        position[:, :] = np.nan
        heatmap = np.zeros(shape=(16, 64, 64), dtype=np.float32)

        keypoints = annorect.annopoints.point

        # Unify the shape.
        if type(keypoints) is not np.ndarray:
            keypoints = [keypoints]

        flip = random() <= 0.4
        if self.augment and self.task == Task.Train and flip:
            for idx in range(len(keypoints)):
                keypoints[idx].x = 2 * center.x - keypoints[idx].x
                keypoints[idx].id = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10][keypoints[idx].id]
            rotate = -rotate
            image = transforms.functional.hflip(image)

        for keypoint in keypoints:
            joint = keypoint.id

            in_image = Vector2(keypoint.x, keypoint.y)
            position[joint] = [in_image.x, in_image.y]

            in_heatmap = (in_image - center) * 64 / (200 * scale)

            if rotate != 0:
                cos = math.cos(rotate * math.pi / 180)
                sin = math.sin(rotate * math.pi / 180)
                in_heatmap = Vector2(sin * in_heatmap.y + cos * in_heatmap.x, cos * in_heatmap.y - sin * in_heatmap.x)

            in_heatmap = in_heatmap + Vector2(64 // 2, 64 // 2)

            if min(in_heatmap) < 0 or max(in_heatmap) >= 64:
                continue

            heatmap[joint, :, :] = draw_heatmap(64, in_heatmap.y, in_heatmap.x)

        if self.task == 'train' and self.augment:
            image = self.color_jitter(image)

        return self.transform(image), heatmap, position, np.asarray([center.x, center.y]), scale, np.asarray([head])

    def __len__(self):
        return len(self.subset)
