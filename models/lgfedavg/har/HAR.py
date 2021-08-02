import torch.nn as nn
import torch.nn.functional as F


class HAR(nn.Module):
    def __init__(self):
        super(HAR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Flatten()
        )

        self.clf = nn.Sequential(
            nn.Linear(1184, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        return x


if __name__ == '__main__':
    import torch
    model = HAR()
    _x = torch.rand((50, 1, 561))
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
