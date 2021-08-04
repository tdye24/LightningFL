import torch.nn as nn


class CIFAR10(nn.Module):
    def __init__(self, dropout):
        super(CIFAR10, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[0]),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[1]),
            nn.Flatten()
        )

        self.clf = nn.Sequential(
            nn.Linear(5 * 5 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout[2]),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # 50x3x32x32
        x = self.conv(x)
        x = self.clf(x)
        return x


if __name__ == '__main__':
    import torch
    model = CIFAR10(dropout=[0.25, 0.25, 0.5])
    _x = torch.rand((50, 3, 32, 32))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    total = 0
    keys = list(model.state_dict().keys())
    keys = list(reversed(keys))  # [top -> down(near the data)]
    print(keys)
    for (key, value) in model.named_parameters():
        if keys.index(key) < 2 * 2:
            total += value.numel()
    print("Comm. Parameters {}".format(total))