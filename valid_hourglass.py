import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

import MPII
import model.hourglass
from util import config
from util.visualize import colorize, overlap

hourglass, optimizer, step, train_epoch = model.hourglass.load(config.hourglass.parameter_dir, config.hourglass.device)
criterion = nn.MSELoss()

# train_epoch equals -1 means that training is over
if train_epoch != -1:

    # Reset statistics of batch normalization
    hourglass.reset_statistics()
    hourglass.train()

    train_loader = DataLoader(
        MPII.Dataset(
            root=config.hourglass.data_dir,
            task='train',
            augment=False,
        ),
        batch_size=config.hourglass.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.hourglass.num_workers,
    )

    # Compute statistics of batch normalization from the train subset
    with tqdm(total=len(train_loader), desc='%d epoch' % train_epoch) as progress:
        with torch.set_grad_enabled(False):
            for images, _, _, _, _, _ in train_loader:
                images = images.to(config.hourglass.device)
                outputs = hourglass(images)

                progress.update(1)

    # epoch equals -1 means that training is over
    epoch = -1
    torch.save(
        {
            'epoch': epoch,
            'step': step,
            'state': hourglass.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        '{parameter_dir}/{epoch}.save'.format(parameter_dir=config.hourglass.parameter_dir, epoch=epoch)
    )

hourglass = hourglass.eval()

valid_data = DataLoader(
    MPII.Dataset(
        root=config.hourglass.data_dir,
        task='valid',
        augment=False,
    ),
    batch_size=config.hourglass.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.hourglass.num_workers,
)

total = torch.zeros((14,)).int()
hit = torch.zeros((14,)).int()

writer = SummaryWriter(log_dir=config.hourglass.log_dir)
resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=[64, 64]),
    transforms.ToTensor(),
])

with tqdm(total=len(valid_data), desc='%d epoch' % train_epoch) as progress:
    with torch.set_grad_enabled(False):
        for images, heatmaps, keypoints, centers, scales, heads in valid_data:
            images = images.to(config.hourglass.device)
            heatmaps = heatmaps.to(config.hourglass.device)
            centers = centers.to(config.hourglass.device).float()
            scales = scales.to(config.hourglass.device).float()

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

            if step % 10 == 0:
                images = torch.stack([resize(image) for image in images.cpu()]).to(config.hourglass.device)

                ground_truth = overlap(images=images, heatmaps=colorize(heatmaps))
                prediction = overlap(images=images, heatmaps=colorize(outputs))

                writer.add_image('ground truth', ground_truth.data, step)
                writer.add_image('prediction', prediction.data, step)

            dists = poses - keypoints.to(config.hourglass.device).float()
            dists = torch.sqrt(torch.sum(dists * dists, dim=-1))

            PCKh_temp = dists / heads.view(n_batch, 1).to(config.hourglass.device).float()
            PCKh_pred = torch.zeros((n_batch, 14,))

            PCKh_pred[:, 0:6] = PCKh_temp[:, 0:6]
            PCKh_pred[:, 6:12] = PCKh_temp[:, 10:16]
            PCKh_pred[:, 12:14] = PCKh_temp[:, 8:10]

            temp = (PCKh_pred <= 0.5).float()

            total = total + torch.sum((~torch.isnan(PCKh_pred)), dim=0).int()
            hit = hit + torch.sum(temp, dim=0).int()

            progress.update(1)
            step = step + 1

print(hit.float() / total.float() * 100)
print(hit)
print(total)

writer.close()

# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
#
# import MPII
# import Model.hourglass
# from util import config
#
# assert config.task == 'valid'
#
# data = DataLoader(
#     MPII.Dataset(
#         root=config.root['MPII'],
#         task=config.task,
#     ),
#     batch_size=config.batch_size,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=config.num_workers,
# )
#
# device = torch.device(config.device)
# hourglass, optimizer, criterion, step, pretrained_epoch = Model.hourglass.load_model(device,
#                                                                                      config.pretrained['hourglass'])
#
#
# for key, value in hourglass.state_dict().items():
#     if 'running_mean' in key:
#
#         layer = hourglass
#         modules = key.split('.')[:-1]
#         for module in modules:
#             if module.isdigit():
#                 layer = layer[int(module)]
#             else:
#                 layer = getattr(layer, module)
#         layer.reset_running_stats()
#         layer.momentum = None
#
#
# train_loader = DataLoader(
#     MPII.Dataset(
#         root=config.root['MPII'],
#         task='train',
#         augment=False,
#     ),
#     batch_size=config.batch_size,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=config.num_workers,
# )
#
# hourglass.train()
#
# with tqdm(total=len(train_loader), desc='%d epoch' % pretrained_epoch) as progress:
#
#     with torch.set_grad_enabled(False):
#
#         for images, heatmaps, keypoints in train_loader:
#             images_cpu = images
#             images = images.to(device)
#             heatmaps = heatmaps.to(device)
#
#             optimizer.zero_grad()
#             outputs = hourglass(images)
#
#             progress.update(1)
#
# hourglass = hourglass.eval()
#
# total = torch.zeros(hourglass.joints).cuda()
# hit = torch.zeros(hourglass.joints).cuda()
#
# with tqdm(total=len(data), desc='%d epoch' % pretrained_epoch) as progress:
#     with torch.set_grad_enabled(False):
#         for images, heatmaps, keypoints in data:
#
#             images_device = images.to(device)
#
#             outputs = hourglass(images_device)
#             outputs = outputs[-1]  # Heatmaps from the last stack in batch-channel-height-width shape.
#
#             n_batch = outputs.shape[0]
#
#             # joint_map = [13, 12, 14, 11, 15, 10, 3, 2, 4, 1, 5, 0, 6, 7, 8, 9]
#             joint_map = [x for x in range(16)]
#
#             for batch in range(n_batch):
#                 for joint, heatmap in enumerate(outputs[batch]):
#
#                     # The empty heatmap means not-annotated.
#                     if np.count_nonzero(heatmaps[batch][joint]) == 0:
#                         continue
#
#                     total[joint] = total[joint] + 1
#
#                     pose = torch.argmax(heatmap.view(-1))
#                     pose = torch.Tensor([int(pose) % 64, int(pose) // 64])
#
#                     dist = pose - keypoints[batch][joint_map[joint]]
#                     dist = torch.sqrt(torch.sum(dist * dist))
#
#                     if torch.le(dist, 64 * 0.1):
#                         hit[joint] = hit[joint] + 1
#
#             progress.update(1)
#
# print(hit)
# print(total)
# print(hit / total * 100)  # In percentage.
