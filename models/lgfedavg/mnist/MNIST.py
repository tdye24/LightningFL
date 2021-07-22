import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.clf = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        return x


if __name__ == '__main__':
    model = MNIST()
    x = torch.rand((50, 1, 28, 28))
    output = model(x)
    print(f'{x.shape}->{output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    total = 0
    for key, param in model.named_parameters():
        if key.startswith('shared'):
            total += param.numel()
    print("Comm. Parameters {}".format(total))
