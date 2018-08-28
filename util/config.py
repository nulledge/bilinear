import torch
from dotmap import DotMap

hourglass = DotMap({
    'batch_size': 8,
    'num_workers': 8,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'visualize': True,
    'parameter_dir': 'parameter/hourglass',
    'log_dir': 'log/hourglass',
    'data_dir': 'data/MPII',
    'prediction_dir': '/media/nulledge/2nd/ubuntu/bilinear/prediction',
})
