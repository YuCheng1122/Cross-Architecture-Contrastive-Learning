import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)
    
class Adapter(nn.Module):
    def __init__(self, dim=256, num_blocks=2):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(dim, dim)  

    def forward(self, x):
        out = self.blocks(x)
        return self.fc_out(out)


class LSTMAdapter(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        x = x.unsqueeze(1) 
        out, (h_n, c_n) = self.lstm(x) 
        last_hidden = h_n[-1]     
        return self.fc(last_hidden)   
    
class MLPAdapter(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.mlp(x)




