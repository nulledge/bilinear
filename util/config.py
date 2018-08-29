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

bilinear = DotMap({
    'batch_size': 64,
    'num_workers': 8,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'visualize': True,
    'parameter_dir': 'parameter/bilinear',
    'log_dir': 'log/bilinear',
    'data_dir': 'data/Human3.6M',
    'lr_decay': {
        'activate': True,
        'condition': (lambda step: step % 100000 == 0 or step == 1),
        'function': (lambda step: 1.0e-3 * 0.96 ** (step / 100000)),
    },
})
