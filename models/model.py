import torch.nn as nn
import torchvision.models as models


class MyModel(nn.Module):
    def __init__(self, hidden_units, out_size, dropout=0.1, pretrained=True):
        super(MyModel, self).__init__()

        self.resnet = models.resnet18(pretrained=pretrained)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_units, out_size),
        )

    def forward(self, x):
        out = self.resnet(x)
        return out


def init_nets(n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net = MyModel(args.hidden_dim, args.out_dim)
        nets[net_i] = net

    return nets
