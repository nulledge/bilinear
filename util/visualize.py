import numpy as np
import skimage.transform
import torch
from visdom import Visdom

viz = Visdom()


def merge_to_color_heatmap(batch_heatmaps, h_format='NCHW'):
    Color = torch.cuda.FloatTensor(
        [[0, 0, 0.5],
         [0, 0, 1],
         [0, 1, 0],
         [1, 1, 0],
         [1, 0, 0]]
    )

    if h_format == 'NHWC':
        batch_heatmaps = batch_heatmaps.permute(0, 3, 1, 2).contiguous()

    batch, joints, height, width = batch_heatmaps.size()

    heatmaps = batch_heatmaps.clamp(0, 1.).view(-1)

    frac = torch.div(heatmaps, 0.25)
    lower_indices, upper_indices = torch.floor(frac).long(), torch.ceil(frac).long()

    t = frac - torch.floor(frac)
    t = t.view(-1, 1)

    k = Color.index_select(0, lower_indices)
    k_1 = Color.index_select(0, upper_indices)

    color_heatmap = (1.0 - t) * k + t * k_1
    color_heatmap = color_heatmap.view(batch, joints, height, width, 3)
    color_heatmap = color_heatmap.permute(0, 4, 2, 3, 1)
    color_heatmap, _ = torch.max(color_heatmap, 4)

    return color_heatmap


def draw_line(x, y, window):
    assert viz.check_connection()

    return viz.line(X=x,
                    Y=y,
                    win=window,
                    update='append' if window is not None else None)


def draw_merged_image(heatmaps, images):
    heatmaps = merge_to_color_heatmap(heatmaps.data)
    heatmaps = heatmaps.permute(0, 2, 3, 1).cpu()  # NHWC

    resized_heatmaps = list()
    for idx, ht in enumerate(heatmaps):
        color_ht = skimage.transform.resize(ht.numpy(), (256, 256), mode='constant')
        resized_heatmaps.append(color_ht.transpose(2, 0, 1))

    resized_heatmaps = np.stack(resized_heatmaps, axis=0)

    images = images * 0.6
    overlayed_image = np.clip(images + resized_heatmaps * 0.4, 0, 1.)

    return overlayed_image

    # return viz.images(tensor=overlayed_image, nrow=4, win=window)


def draw(step, loss, images, heatmaps, outputs, windows):
    loss_window, gt_image_window, out_image_window = windows

    if step % 10 == 0:
        out = outputs[-1, :].squeeze().contiguous()
        gt_images = images.cpu().numpy()
        gt_image_window = draw_merged_image(out, gt_images.copy(), gt_image_window)
        out_image_window = draw_merged_image(heatmaps, gt_images.copy(), out_image_window)

    loss_window = draw_line(x=step,
                            y=np.array([float(loss.data)]),
                            window=loss_window)

    return [loss_window, gt_image_window, out_image_window]
