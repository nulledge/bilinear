import numpy as np
import scipy.io
import torch
import torch.utils.data as torch_data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from vectormath import Vector2
from tqdm import tqdm

import MPII
import Model.hourglass
from util import config
from MPII.util import crop_image

assert config.task == 'eval'

device = torch.device(config.device)
hourglass, optimizer, criterion, step, pretrained_epoch = Model.hourglass.load_model(device,
                                                                                     config.pretrained['hourglass'])

if pretrained_epoch != -1:

    for key, value in hourglass.state_dict().items():
        if 'running_mean' in key:

            layer = hourglass
            modules = key.split('.')[:-1]
            for module in modules:
                if module.isdigit():
                    layer = layer[int(module)]
                else:
                    layer = getattr(layer, module)
            layer.reset_running_stats()
            layer.momentum = None

    train_loader = DataLoader(
        MPII.Dataset(
            root=config.root['MPII'],
            task='train',
            augment=False,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
    )

    hourglass.train()

    with tqdm(total=len(train_loader), desc='%d epoch' % pretrained_epoch) as progress:
        with torch.set_grad_enabled(False):
            for images, heatmaps, keypoints in train_loader:
                images_cpu = images
                images = images.to(device)
                heatmaps = heatmaps.to(device)

                optimizer.zero_grad()
                outputs = hourglass(images)

                progress.update(1)

    torch.save(
        {
            'epoch': -1,
            'step': step,
            'state': hourglass.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        '{pretrained}/{epoch}.save'.format(pretrained=config.pretrained['hourglass'], epoch=-1)
    )

hourglass = hourglass.eval()


class EvalData(torch_data.Dataset):

    def __init__(self):
        anno = '/media/nulledge/3rd/data/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
        anno = scipy.io.loadmat(anno, squeeze_me=True, struct_as_record=False)['RELEASE']

        test_subset = np.where(anno.img_train == 0)
        annolist = anno.annolist[test_subset]
        single_person = anno.single_person[test_subset]

        self.test_data = []

        for img_idx in range(len(annolist)):
            rect = annolist[img_idx].annorect

            # Convert scalar to array.
            if not isinstance(rect, np.ndarray):
                rect = [rect]

            if not isinstance(single_person[img_idx], np.ndarray):
                single_person[img_idx] = [single_person[img_idx]]

            for r_idx in range(len(rect)):
                try:
                    # Python is 0-based but MPII r_idx is 1-based.
                    if r_idx + 1 not in single_person[img_idx]:
                        continue

                    center = (rect[r_idx].objpos.x, rect[r_idx].objpos.y)
                    scale = rect[r_idx].scale
                    img_name = annolist[img_idx].image.name

                    # Python is 0-based but MPII img_idx and r_idx is 1-based.
                    self.test_data.append({
                        'center': center,
                        'scale': scale,
                        'img_name': img_name,
                        'img_idx': img_idx + 1,
                        'r_idx': r_idx + 1,
                    })

                except Exception as e:
                    pass
        self.to_tensor = ToTensor()

    def __getitem__(self, index):

        data = self.test_data[index]

        center = Vector2(data['center'][0], data['center'][1])
        scale = data['scale'] * 1.25
        rotate = 0
        img_name = data['img_name']

        img_idx = data['img_idx']
        r_idx = data['r_idx']

        image_path = '{image_path}/{image_name}'.format(image_path='/media/nulledge/3rd/data/MPII/images',
                                                        image_name=img_name)
        image = crop_image(image_path, center, scale, rotate)

        return self.to_tensor(image), np.asarray([center.x, center.y], dtype=np.float32), scale, img_idx, r_idx

    def __len__(self):
        return len(self.test_data)


test_data = torch_data.DataLoader(
    EvalData(),
    batch_size=8,
    shuffle=False,
    pin_memory=True,
    num_workers=8,
)

with tqdm(total=len(test_data), desc='%d epoch' % pretrained_epoch) as progress:
    with torch.set_grad_enabled(False):
        for images, centers, scales, img_idxs, r_idxs in test_data:
            images = images.to(device)
            centers = centers.to(device).float()
            scales = scales.to(device).float()
            outputs = hourglass(images)
            outputs = outputs[-1]  # Heatmaps from the last stack in batch-channel-height-width shape.

            n_batch = outputs.shape[0]

            poses = torch.argmax(outputs.view(n_batch, 16, -1), dim=-1)
            poses = torch.stack([
                poses % 64,
                poses // 64,
            ], dim=-1).float()
            poses = poses - 32
            poses = centers.view(n_batch, 1, 2) + poses / 64 * scales.view(n_batch, 1, 1) * 200

            for batch, img_idx, r_idx in zip(range(n_batch), img_idxs, r_idxs):
                with open('/media/nulledge/2nd/ubuntu/bilinear/prediction/{img_idx}.{r_idx}.txt'.format(img_idx=img_idx,
                                                                                                        r_idx=r_idx),
                          'w') as f:
                    pose = poses[batch]
                    for joint in range(16):
                        f.writelines('{joint} {x} {y}\n'.format(joint=joint, x=pose[joint, 0], y=pose[joint, 1]))

            progress.update(1)

