import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.shared_encoder = nn.Sequential(
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

        self.private_encoder = nn.Sequential(
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

        self.clf = nn.Sequential(
            nn.Linear(64 * 5 * 5 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )

        self.critic = nn.Sequential(
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        gFeature = self.shared_encoder(x)
        lFeature = self.private_encoder(x)
        feature = torch.cat((gFeature, lFeature), dim=-1)
        gValue = self.critic(gFeature)
        lValue = self.critic(lFeature)
        out = self.clf(feature)
        return gFeature, lFeature, gValue, lValue, out

    def metaCritic(self, x):
        return self.critic(x)


if __name__ == '__main__':
    model = CIFAR10()
    x = torch.rand((50, 3, 32, 32))
    gFeature_, lFeature_, gValue_, lValue_, output_ = model(x)
    print(f'{x.shape}->gFeature_{gFeature_.shape}')
    print(f'{x.shape}->lFeature_{lFeature_.shape}')
    print(f'{x.shape}->gValue{gValue_.shape}')
    print(f'{x.shape}->lValue{lValue_.shape}')
    print(f'{x.shape}->output{output_.shape}')

    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    total = 0
    for key, param in model.named_parameters():
        if key.startswith('shared') or key.startswith('critic'):
            total += param.numel()
    print("Comm. Parameters {}".format(total))
