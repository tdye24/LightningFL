import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10(nn.Module):
    def __init__(self, dropout):
        super(CIFAR10, self).__init__()
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[0]),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[1]),
            nn.Flatten()
        )

        self.private_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[2]),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[4]),
            nn.Flatten()
        )

        self.clf = nn.Sequential(
            nn.Linear(64 * 5 * 5 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout[5]),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        gFeature = self.shared_encoder(x)
        lFeature = self.private_encoder(x)
        feature = torch.cat((gFeature, lFeature), dim=-1)
        out = self.clf(feature)
        return out


if __name__ == '__main__':
    model = CIFAR10()
    x = torch.rand((50, 3, 32, 32))
    output_ = model(x)
    print(f'{x.shape}->output{output_.shape}')

    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    total = 0
    for key, param in model.named_parameters():
        if key.startswith('shared') or key.startswith('critic'):
            total += param.numel()
    print("Comm. Parameters {}".format(total))
