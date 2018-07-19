import MPII
import numpy as np
import imageio
from torch.utils.data import DataLoader
from tqdm import tqdm


data = DataLoader(
    MPII.Dataset(
        root='/media/nulledge/2nd/data/MPII/',
        task='train',
    ),
    batch_size=8,
    shuffle=True,
    pin_memory=True,
    num_workers=8
)

num = 0
with tqdm(total=len(data)) as progress:
    for images, heatmaps, keypoints in data:

        for image, heatmap, keypoint in zip(images, heatmaps, keypoints):

            for x in range(64):
                for y in range(64):
                    heatmap[0, y, x] = max(heatmap[:, y, x])

            imageio.imwrite('%d-rgb.jpg' % num, np.asarray(image.permute(1, 2, 0)))
            imageio.imwrite('%d-heat.jpg' % num, np.asarray(heatmap[0, :, :]))

            num = num + 1

        progress.update(1)

        break