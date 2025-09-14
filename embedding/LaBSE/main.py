import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


# --- Encoder (MLP) ---
class MLPEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, out_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.fc(x), dim=-1)


# --- Dual Encoder ---
class DualEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, out_dim=256, margin=0.3, scale=10):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim, out_dim)
        self.margin = margin
        self.scale = scale

    def forward(self, src_vecs, tgt_vecs):
        src_emb = self.encoder(src_vecs)
        tgt_emb = self.encoder(tgt_vecs)

        logits = torch.matmul(src_emb, tgt_emb.T) * self.scale
        batch_size = src_vecs.size(0)

        # AMS
        margin_mask = torch.eye(batch_size, device=src_vecs.device)
        logits = logits - self.margin * margin_mask

        labels = torch.arange(batch_size, device=src_vecs.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_j = F.cross_entropy(logits.T, labels)
        loss = (loss_i + loss_j) / 2
        return loss, src_emb, tgt_emb


# --- Cosine similarity 評估 ---
def evaluate_similarity(model, loader, device):
    model.eval()
    sims = []
    with torch.no_grad():
        for arm_vecs, x86_vecs in loader:
            arm_vecs, x86_vecs = arm_vecs.to(device), x86_vecs.to(device)
            _, src_emb, tgt_emb = model(arm_vecs, x86_vecs)

            # 正樣本：對應 pair
            pos_sim = F.cosine_similarity(src_emb, tgt_emb).mean().item()
            sims.append(pos_sim)
    return np.mean(sims)


# --- Load Data ---
data_path = "/home/tommy/Projects/pcodeFcg/dataset/alignment_vector/train_arm_vector_Os.pickle"
with open(data_path, "rb") as f:
    data = pickle.load(f)

ARM = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
X86 = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32)

dataset = TensorDataset(ARM, X86)

# 切 90% train / 10% val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")

# --- Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualEncoder(input_dim=ARM.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for arm_vecs, x86_vecs in train_loader:
        arm_vecs, x86_vecs = arm_vecs.to(device), x86_vecs.to(device)

        loss, _, _ = model(arm_vecs, x86_vecs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 計算 validation cosine similarity
    val_sim = evaluate_similarity(model, val_loader, device)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss = {total_loss/len(train_loader):.4f}, "
          f"Val CosSim = {val_sim:.4f}")

# --- 存模型 ---
save_path = "dual_encoder_alignment.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved at {save_path}")
