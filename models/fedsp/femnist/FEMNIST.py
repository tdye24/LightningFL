import torch
import torch.nn as nn


class FEMNIST(nn.Module):
    def __init__(self, dropout):
        super(FEMNIST, self).__init__()
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[0]),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[1]),
            nn.Flatten()
        )

        self.private_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[2]),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[3]),
            nn.Flatten()
        )

        self.clf = nn.Sequential(
            nn.Linear(64*7*7*2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout[4]),
            nn.Linear(1024, 62)
        )

    def forward(self, x):
        x = x.reshape((-1, 1, 28, 28))
        gFeature = self.shared_encoder(x)
        lFeature = self.private_encoder(x)
        feature = torch.cat((gFeature, lFeature), dim=-1)
        out = self.clf(feature)
        return out


if __name__ == '__main__':
    model = FEMNIST(dropout=[0.25, 0.25, 0.75, 0.75, 0.9])
    _x = torch.rand((50, 1, 28, 28))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

