import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 4, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(4, 16, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(16, 64, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.cls = nn.Sequential(
            nn.Linear(64 * 6 * 6, 400),
            nn.ReLU(),
            nn.Linear(400, 10)
        )

    def forward(self, x):
        feature = self.features(x)
        return self.cls(feature.view(x.size(0), -1))
