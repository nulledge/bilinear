import math
import numpy as np
import os
import scipy, scipy.io
import skimage, skimage.io
import torch
import torch.utils.data as torch_data
from random import random, shuffle, uniform
from torchvision import transforms
from vectormath import Vector2

from .util import rand, crop_image, draw_heatmap


class Dataset(torch_data.Dataset):

    # ID_TO_JOINT = {
    #     0: JOINT.R_Ankle,
    #     1: JOINT.R_Knee,
    #     2: JOINT.R_Hip,
    #     3: JOINT.L_Hip,
    #     4: JOINT.L_Knee,
    #     5: JOINT.L_Ankle,
    #     6: JOINT.M_Pelvis,
    #     7: JOINT.M_Thorax,
    #     8: JOINT.M_UpperNeck,
    #     9: JOINT.M_HeadTop,
    #     10: JOINT.R_Wrist,
    #     11: JOINT.R_Elbow,
    #     12: JOINT.R_Shoulder,
    #     13: JOINT.L_Shoulder,
    #     14: JOINT.L_Elbow,
    #     15: JOINT.L_Wrist
    # }

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
        if self.task == 'train' and self.augment:
            scale = scale * 2 ** rand(0.25)
            rotate = rand(30) if random() <= 0.4 else 0.0

        objpos = annorect.objpos
        center = Vector2(objpos.x, objpos.y)
        center.setflags(write=False)

        image_name = annolist[img_idx].image.name
        image_path = '{image_path}/{image_name}'.format(image_path=self.image_path, image_name=image_name)
        image = skimage.img_as_float(skimage.io.imread(image_path))
        image = crop_image(image, center, scale, rotate, resolution=256)

        position = np.zeros(shape=(16, 2), dtype=np.float32)
        heatmap = np.zeros(shape=(16, 64, 64), dtype=np.float32)

        keypoints = annorect.annopoints.point

        # Unify the shape.
        if type(keypoints) is not np.ndarray:
            keypoints = [keypoints]

        for keypoint in keypoints:
            joint = keypoint.id
            in_image = Vector2(keypoint.x, keypoint.y)
            in_heatmap = (in_image - center) * 64 / (200 * scale)

            if rotate != 0:
                cos = math.cos(rotate * math.pi / 180)
                sin = math.sin(rotate * math.pi / 180)
                in_heatmap = Vector2(sin * in_heatmap.y + cos * in_heatmap.x, cos * in_heatmap.y - sin * in_heatmap.x)

            in_heatmap = in_heatmap + Vector2(64 // 2, 64 // 2)
            position[joint] = [in_heatmap.x, in_heatmap.y]

            if min(in_heatmap) < 0 or max(in_heatmap) >= 64:
                continue

            heatmap[joint, :, :] = draw_heatmap(64, in_heatmap.y, in_heatmap.x)

        image = image.astype(dtype=np.float32)
        if self.task == 'train' and self.augment:
            image[:, :, 0] *= uniform(0.6, 1.4)
            image[:, :, 1] *= uniform(0.6, 1.4)
            image[:, :, 2] *= uniform(0.6, 1.4)
            image = np.clip(image, 0, 1)

        return image.transpose(2, 0, 1), heatmap, position

    def __len__(self):
        return len(self.subset)
