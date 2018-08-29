import torch
import torch.nn as nn
import numpy as np
import os


def heavy_linear(in_features, out_features, bias=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p=0.5),
    )


class BilinearUnit(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = heavy_linear(in_features=2 * (17 - 1), out_features=1024)
        self.bilinear = nn.ModuleList([
            nn.Sequential(
                heavy_linear(in_features=1024, out_features=1024),
                heavy_linear(in_features=1024, out_features=1024),
            ) for _ in range(2)
        ])
        self.decode = nn.Linear(in_features=1024, out_features=3 * (17 - 1), bias=True)

    def forward(self, in_tensor):
        out_tensor = in_tensor

        out_tensor = self.encode(out_tensor)
        for bilinear in self.bilinear:
            skip_tensor = out_tensor
            out_tensor = bilinear(out_tensor)
            out_tensor = out_tensor + skip_tensor
        out_tensor = self.decode(out_tensor)

        return out_tensor

    def reset_statistics(self):
        for key, value in self.state_dict().items():
            if 'running_mean' in key:

                layer = self
                modules = key.split('.')[:-1]
                for module in modules:
                    if module.isdigit():
                        layer = layer[int(module)]
                    else:
                        layer = getattr(layer, module)
                layer.reset_running_stats()
                layer.momentum = None


def load(parameter_dir, device, learning_rate=1.0e-3):
    bilinear = BilinearUnit().to(device)
    optimizer = torch.optim.Adam(bilinear.parameters(), lr=learning_rate)
    step = 1

    epoch_to_load = 0
    for _, _, files in os.walk(parameter_dir):
        for file in files:
            # The name of parameter file is {epoch}.save
            name, extension = file.split('.')
            epoch = int(name)

            if epoch > epoch_to_load:
                epoch_to_load = epoch

    if epoch_to_load != 0:
        parameter_file = '{parameter_dir}/{epoch}.save'.format(parameter_dir=parameter_dir, epoch=epoch_to_load)
        parameter = torch.load(parameter_file)

        bilinear.load_state_dict(parameter['state'])
        optimizer.load_state_dict(parameter['optimizer'])
        step = parameter['step']

    else:
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)

        bilinear.apply(weight_init)  # (lambda x: nn.init.kaiming_normal(x.weight)))

    return bilinear, optimizer, step, epoch_to_load
