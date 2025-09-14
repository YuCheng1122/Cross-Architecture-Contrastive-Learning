import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x

class LSTM(nn.Module):
    def __init__(self, dim=256, hidden_dim=256, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, dim)
    def forward(self, x):
        if x.dim() == 2:  
            x = x.unsqueeze(1)
        out, (h, c) = self.lstm(x) 
        last_out = out[:, -1, :]
        return self.fc(last_out)
    
class SimSiam(nn.Module):
    def __init__(self, base_encoder, dim=256, pred_dim=128):
        super(SimSiam, self).__init__()

        # backbone -> 輸入 (batch, 256)，輸出 (batch, 256)
        self.encoder = base_encoder(dim=dim)

        prev_dim = dim

        # projector (3-layer MLP)
        self.projector = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)
        )

        # predictor (2-layer MLP)
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        z1 = self.projector(f1)
        p1 = self.predictor(z1)

        f2 = self.encoder(x2)
        z2 = self.projector(f2)
        p2 = self.predictor(z2)

        return (z1, p1), (z2, p2)

