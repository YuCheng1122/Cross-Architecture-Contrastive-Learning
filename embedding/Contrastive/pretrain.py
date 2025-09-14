import pickle
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from model import SimSiam, MLP


def simsiam_loss(p, z):
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def load_data(path, batch_size=64):
    with open(path, "rb") as f:
        data = pickle.load(f)  # list of (vec1, vec2, label)

    v1 = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
    v2 = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32)
    labels = torch.tensor(np.array([d[2] for d in data]), dtype=torch.long)

    dataset = TensorDataset(v1, v2, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for v1, v2, _ in train_loader:
        v1, v2 = v1.to(device), v2.to(device)

        (z1, p1), (z2, p2) = model(v1, v2)
        loss = 0.5 * (simsiam_loss(p1, z2) + simsiam_loss(p2, z1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


if __name__ == "__main__":
    data_path = "/home/tommy/Projects/pcodeFcg/dataset/alignment_vector/train_arm_vector.pickle"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_loss = float("inf")
    train_losses = []

    # load data
    train_loader = load_data(data_path, batch_size=64)

    # init model
    model = SimSiam(base_encoder=MLP, dim=256, pred_dim=128).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

    for epoch in range(200):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "simsiam_pretrain_best.pth")
            print(f"Saved new best model at epoch {epoch+1} (loss={train_loss:.4f})")


    # plot loss curve
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("train_loss_curve.png")
