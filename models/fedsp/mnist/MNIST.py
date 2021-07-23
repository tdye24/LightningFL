import torch
import torch.nn as nn


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.shared_encoder = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.private_encoder = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.clf = torch.nn.Sequential(
            torch.nn.Linear(64*7*7*2, 512),  # 乘2因为global_feat和local_feat拼在一起
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 10)
        )

    def forward(self, x):
        gFeature = self.shared_encoder(x)
        lFeature = self.private_encoder(x)

        feature = torch.cat((gFeature, lFeature), dim=-1)
        output = self.clf(feature)
        return output


if __name__ == '__main__':
    model = MNIST()
    _x = torch.rand((50, 1, 28, 28))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    total = 0
    for key, param in model.named_parameters():
        if key.startswith('shared'):
            total += param.numel()
    print("Comm. Parameters {}".format(total))