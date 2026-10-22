import torch
import torch.nn as nn

class SoftPenaltyPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 150), nn.Tanh(),
            nn.Linear(150, 150), nn.Tanh(),
            nn.Linear(150, 150), nn.Tanh(),
            nn.Linear(150, 150), nn.Tanh(),
            nn.Linear(150, 150), nn.Tanh(),
            nn.Linear(150, 150), nn.Tanh(),
            nn.Linear(150, 150), nn.Tanh(),
            nn.Linear(150, 3)
        )

        self.e15_pred = nn.Parameter(torch.tensor([0.10], dtype=torch.float32))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x_tensor):
        out = self.net(x_tensor)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]
