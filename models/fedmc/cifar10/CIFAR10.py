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

        self.shared_decoder = nn.Sequential(
            nn.Linear(64 * 5 * 5, 3 * 32 * 32)
        )
        self.private_decoder = nn.Sequential(
            nn.Linear(64 * 5 * 5, 3 * 32 * 32)
        )

        self.clf = nn.Sequential(
            nn.Linear(3 * 32 * 32 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )

        self.critic = nn.Sequential(
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
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        gFeature = self.shared_decoder(self.shared_encoder(x))
        lFeature = self.private_decoder(self.private_encoder(x))
        feature = torch.cat((gFeature, lFeature), dim=-1)
        gFeature = gFeature.view(-1, 3, 32, 32)
        lFeature = lFeature.view(-1, 3, 32, 32)
        gValue = self.critic(gFeature)
        lValue = self.critic(lFeature)
        out = self.clf(feature)
        return gFeature, lFeature, gValue, lValue, out

    def metaCritic(self, x):
        return self.critic(x)


if __name__ == '__main__':
    model = CIFAR10()
    x = torch.rand((50, 3, 32, 32))
    output, gD, lD = model(x)
    print(f'{x.shape}->{output.shape}')
    print(f'{x.shape}->g_critic_out{gD.shape}')
    print(f'{x.shape}->l_critic_out{lD.shape}')

    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    total = 0
    for key, param in model.named_parameters():
        if key.startswith('shared') or key.startswith('critic'):
            total += param.numel()
    print("Comm. Parameters {}".format(total))
