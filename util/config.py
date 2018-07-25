import torch

pretrained = '/media/nulledge/2nd/ubuntu/bilinear/pretrained'
root = {
    'MPII': '/media/nulledge/2nd/data/MPII/',
    'Human3.6M': '/media/nulledge/2nd/data/Human3.6M/converted',
}
task = 'train'
batch_size = 8
num_workers = 8
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
visualize = True
