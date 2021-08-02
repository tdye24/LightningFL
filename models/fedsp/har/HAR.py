import torch
import torch.nn as nn
import torch.nn.functional as F


class HAR(nn.Module):
    def __init__(self):
        super(HAR, self).__init__()
        self.shared_encoder = nn.Sequential(
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

        self.private_encoder = nn.Sequential(
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
            nn.Linear(1184 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        gFeature = self.shared_encoder(x)
        lFeature = self.private_encoder(x)
        feature = torch.cat((gFeature, lFeature), dim=-1)
        out = self.clf(feature)
        return out


if __name__ == '__main__':
    model = HAR()
    x = torch.rand((50, 1, 561))
    output_ = model(x)
    print(f'{x.shape}->output{output_.shape}')

    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    total = 0
    for key, param in model.named_parameters():
        if key.startswith('shared') or key.startswith('critic'):
            total += param.numel()
    print("Comm. Parameters {}".format(total))
