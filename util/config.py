import torch

root = '/media/nulledge/2nd/data/MPII/'
task = 'train'
batch_size = 8
num_workers = 8
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
visualize = True
