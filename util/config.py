import torch

pretrained = {
    'hourglass': '/media/nulledge/2nd/ubuntu/bilinear/pretrained/hourglass',
    'bilinear': '/media/nulledge/2nd/ubuntu/bilinear/pretrained/bilinear',
}
root = {
    'MPII': '/media/nulledge/2nd/data/MPII/',
    'Human3.6M': '/media/nulledge/2nd/data/Human3.6M/converted',
}
task = 'train'
batch_size = 64
num_workers = 8
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
visualize = True
