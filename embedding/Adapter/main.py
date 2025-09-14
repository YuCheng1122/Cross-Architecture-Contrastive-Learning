import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model import Adapter, LSTMAdapter, MLPAdapter

def similarity_score(x1, x2):
    dist = torch.norm(x1 - x2, dim=1)
    return 1 / (1 + dist)

def train_adapter_step(batch, adapter, optimizer, device):
    v1, v2, labels = [b.to(device) for b in batch]
    adapter.train()
    optimizer.zero_grad()

    z1 = adapter(v1)
    z2 = adapter(v2)

    sims = similarity_score(z1, z2)
    loss = F.binary_cross_entropy(sims, labels.float())
    loss.backward()
    optimizer.step()
    return loss.item()


def load_data(path, batch_size=64):
    with open(path, "rb") as f:
        data = pickle.load(f)  # list of (vec1, vec2, label)

    v1 = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
    v2 = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32)
    labels = torch.tensor(np.array([d[2] for d in data]), dtype=torch.long)

    dataset = TensorDataset(v1, v2, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    data_path = "/home/tommy/Projects/pcodeFcg/dataset/alignment_vector/train_arm_vector_posneg.pickle"
    loader = load_data(data_path, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = MLPAdapter(input_dim=256).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    for epoch in range(10):  
        total_loss = 0
        for batch in loader:
            loss = train_adapter_step(batch, adapter, optimizer, device)
            total_loss += loss
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    torch.save(adapter.state_dict(), "ARM_ResNet_adapter_10.pth")
