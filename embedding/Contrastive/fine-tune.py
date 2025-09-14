import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from model import SimSiam, MLP


def load_data(path, batch_size=64):
    with open(path, "rb") as f:
        data = pickle.load(f)  # list of (vec1, vec2, label)

    v1 = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
    labels = torch.tensor(np.array([d[2] for d in data]), dtype=torch.long)

    dataset = TensorDataset(v1, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


class Classifier(nn.Module):
    def __init__(self, backbone, dim=256, num_classes=4, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone.encoder  # 只取 encoder
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.fc(feat)


if __name__ == "__main__":
    data_path = "/home/tommy/Projects/pcodeFcg/dataset/alignment_vector/train_arm_vector.pickle"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_loader = load_data(data_path, batch_size=64)

    # load pretrained simsiam
    simsiam = SimSiam(base_encoder=MLP, dim=256, pred_dim=128).to(device)
    simsiam.load_state_dict(torch.load("simsiam_pretrain_best.pth", map_location=device))

    # build classifier (改 num_classes)
    model = Classifier(simsiam, dim=256, num_classes=4, freeze_backbone=True).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # train loop
    for epoch in range(30):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.4f}")
