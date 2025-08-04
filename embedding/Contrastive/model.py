import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return self.act(out + x)

class Adapter(nn.Module):
    def __init__(self, dim=256, num_blocks=2):
        super().__init__()
        blocks = [ResNet(dim) for _ in range(num_blocks)]
        self.adapter = nn.Sequential(*blocks)

    def forward(self, x):
        return self.adapter(x)

class SimSiamAdapter(nn.Module):
    """
    Adapter + SimSiam projector & predictor
    """
    def __init__(self, dim=256, proj_hidden=2048, pred_hidden=512, num_blocks=2):
        super().__init__()
        # backbone adapter
        self.backbone = Adapter(dim, num_blocks)
        # projector: 2-layer MLP
        self.projector = nn.Sequential(
            nn.Linear(dim, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, dim, bias=False),
            nn.BatchNorm1d(dim),
        )
        # predictor: 2-layer MLP
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, dim),
        )

    def forward(self, x):
        f = self.backbone(x)      # (B, dim)
        z = self.projector(f)     # (B, dim)
        p = self.predictor(z)     # (B, dim)
        return z, p


def simsiam_loss(p, z):
    """
    Negative cosine similarity with stop-gradient on z
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()
