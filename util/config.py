import torch

pretrained = {
    'hourglass': '/media/nulledge/2nd/ubuntu/bilinear/pretrained/hourglass',
    'bilinear': '/media/nulledge/2nd/ubuntu/bilinear/pretrained/bilinear',
}
root = {
    'MPII': '/media/nulledge/3rd/MPII',
    'Human3.6M': '/media/nulledge/3rd/Human3.6M',
}
task = 'valid'
batch_size = 8
num_workers = 8
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
visualize = True
