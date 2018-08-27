import torch

# MSDN 'Heat Map Color Gradients'
COLOR_SPECTRUM = torch.Tensor([
    [0.0, 0.0, 0.5],  # Navy
    [0.0, 0.0, 1.0],  # Blue
    [0.0, 1.0, 0.0],  # Green
    [1.0, 1.0, 0.0],  # Yellow
    [1.0, 0.0, 0.0],  # Red
])
INCANDESCENT = torch.Tensor([
    [0.0, 0.0, 0.0],  # Black
    [0.5, 0.0, 0.0],  # Dark red
    [1.0, 1.0, 0.0],  # Yellow
    [1.0, 1.0, 1.0],  # White
])


def colorize(heatmaps, color_gradient=COLOR_SPECTRUM):
    color_gradient = color_gradient.to(heatmaps.device)

    batch, _, height, width = heatmaps.shape
    heatmaps, _ = heatmaps.max(dim=1)
    heatmaps = heatmaps.view(-1)

    index = heatmaps.mul(len(color_gradient) - 1).clamp(0, len(color_gradient) - 1)
    lower_bound, upper_bound = (index.floor(), index.ceil())
    rate = (index - lower_bound).view(-1, 1)
    heatmaps = color_gradient.index_select(0, lower_bound.long()) * (1 - rate) \
               + color_gradient.index_select(0, upper_bound.long()) * rate

    return heatmaps.view(batch, height, width, 3).permute(0, 3, 1, 2)  # 3 for RGB channel


def overlap(heatmaps, images, ratio=0.5):
    assert 0.0 <= ratio <= 1.0
    return heatmaps * ratio + images * (1 - ratio)
