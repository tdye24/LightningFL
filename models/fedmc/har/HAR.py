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

        self.critic = nn.Sequential(
            nn.Linear(1184, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
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
    model = HAR()
    x = torch.rand((50, 1, 561))
    gFeature_, lFeature_, gValue_, lValue_, output_ = model(x)
    print(f'{x.shape}->gFeature_{gFeature_.shape}')
    print(f'{x.shape}->lFeature_{lFeature_.shape}')
    print(f'{x.shape}->gValue{gValue_.shape}')
    print(f'{x.shape}->lValue{lValue_.shape}')
    print(f'{x.shape}->output{output_.shape}')

    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    share_encoder_total = 0
    for key, param in model.named_parameters():
        if key.startswith('shared'):
            share_encoder_total += param.numel()
    print("Shared Encoder. Parameters {}".format(share_encoder_total))

    critic_total = 0
    for key, param in model.named_parameters():
        if key.startswith('critic'):
            critic_total += param.numel()
    print("Critic. Parameters {}".format(critic_total))

    print("Without periodically training", (share_encoder_total + critic_total))
    print("With periodically training", (share_encoder_total + critic_total) / 2)