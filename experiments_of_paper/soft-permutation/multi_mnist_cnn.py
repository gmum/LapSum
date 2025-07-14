import torch.nn as nn


class DeepNN(nn.Module):
    def __init__(self, l, final_dim=1):
        super(DeepNN, self).__init__()
        self.l = l

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(l * 7 * 7 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, final_dim),
        )

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(-1, 1, self.l * 28, 28)
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x.view(bs, -1)
