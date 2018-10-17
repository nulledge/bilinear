import torch
import pprint


class Config(object):
    def __init__(self,
                 model_fmt,
                 lr,
                 task):
        self.model_fmt = model_fmt
        self.annotation_path = "./Human3.6M/annot"
        self.image_path = "./Human3.6M/images"
        self.ckpt_path = "./pretrained/"
        # self.pretrained_path = "./test/"
        self.subjects = [1, 5, 6, 7, 8, 9, 11]
        self.task = task
        self.num_parts = 17
        self.heatmap_xy_coefficient = 2
        self.voxel_xy_res = 64
        self.voxel_z_res = [1, 2, 4, 64]
        self.batch = 4
        self.iter = 11000
        self.learning_rate = lr
        self.num_split = 6
        # self.iter = 10
        self.workers = 8
        self.epoch = 100
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        return str(self.__class__) + ": \n" + pprint.pformat(self.__dict__, indent=4)
