import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR100(nn.Module):
    def __init__(self):
        super(CIFAR100, self).__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            nn.Flatten()
        )

        self.private_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            nn.Flatten()
        )

        self.private_clf = nn.Sequential(
            nn.Linear(5 * 5 * 64 * 2, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        # 50x3x32x32
        global_feature = self.shared_conv(x)
        local_feature = self.private_conv(x)
        feature = torch.cat((global_feature, local_feature), dim=-1)
        out = self.private_clf(feature)
        return out


if __name__ == '__main__':
    model = CIFAR100()
    _x = torch.rand((50, 3, 32, 32))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    total = 0
    for key, param in model.named_parameters():
        if key.startswith('shared'):
            total += param.numel()
    print("Comm. Parameters {}".format(total))