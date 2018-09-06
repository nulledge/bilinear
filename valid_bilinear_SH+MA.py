import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import H36M
import model.bilinear
from util import config

data = DataLoader(
    H36M.Dataset(
        data_dir=config.bilinear.data_dir,
        task=H36M.Task.Valid,
        protocol=H36M.Protocol.SH,
    ),
    batch_size=config.bilinear.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.bilinear.num_workers,
)

bilinear, optimizer, step, train_epoch = model.bilinear.load(config.bilinear.parameter_dir, config.bilinear.device)

assert train_epoch == 200

bilinear.eval()

total_dist = dict()
total = dict()

with tqdm(total=len(data), desc='%d epoch' % train_epoch) as progress:
    with torch.set_grad_enabled(False):
        for subset, _, _, action in data:
            in_image_space = subset[H36M.Annotation.Part]
            in_camera_space = subset[H36M.Annotation.S]
            mean = subset[H36M.Annotation.Mean_Of + H36M.Annotation.S]
            stddev = subset[H36M.Annotation.Stddev_Of + H36M.Annotation.S]

            in_image_space = in_image_space.to(config.bilinear.device)
            in_camera_space = in_camera_space.to(config.bilinear.device)
            mean = mean.to(config.bilinear.device)
            stddev = stddev.to(config.bilinear.device)

            n_batch = in_image_space.shape[0]

            # Restore from normalized unit to mm unit
            prediction = bilinear(in_image_space)
            prediction_in_camera_space = stddev * prediction + mean
            in_camera_space = stddev * in_camera_space + mean

            ground_truth = in_camera_space.view(-1, 16, 3)
            prediction = prediction_in_camera_space.view(-1, 16, 3)

            # Compute distance between ground-truth and prediction
            dist = torch.sum(torch.sqrt(torch.sum((prediction - ground_truth) ** 2, dim=2)), dim=1)
            for idx in range(n_batch):

                # Never consider whether sub-action is 1 ({action}_1) or 2 (just {action}
                action[idx] = action[idx].split('_')[0]

                if action[idx] not in total_dist.keys():
                    total_dist[action[idx]] = np.double(0)
                    total[action[idx]] = 0
                total_dist[action[idx]] = total_dist[action[idx]] + np.double(dist[idx].cpu())
                total[action[idx]] = total[action[idx]] + 1

            progress.update(1)

dist = 0.0
cnt = 0
for key, value in total_dist.items():
    print(key, value / (total[key] * 16))
    dist = dist + value
    cnt = cnt + total[key] * 16

print('avg', dist / cnt)