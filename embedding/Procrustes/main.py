import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === 載入資料 ===
data_path = "/home/tommy/Projects/pcodeFcg/dataset/alignment_vector/train_arm_vector.pickle"

with open(data_path, "rb") as f:
    data = pickle.load(f)  # list of (vec1, vec2, label)

# vec1 = ARM, vec2 = X86
ARM = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
X86 = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32)

dataset = TensorDataset(ARM, X86)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"ARM shape: {ARM.shape}, X86 shape: {X86.shape}")


# === 定義模型 (簡單線性轉換) ===
class AlignmentModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)  # 只學一個 W

    def forward(self, x):
        return self.linear(x)

# 維度假設跟你的向量一致
dim = ARM.shape[1]
model = AlignmentModel(dim)

# === 訓練 ===
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    total_loss = 0
    for arm_vec, x86_vec in loader:
        optimizer.zero_grad()
        pred = model(arm_vec)
        loss = criterion(pred, x86_vec)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss = {total_loss/len(loader):.6f}")

# === 測試效果 (cosine similarity) ===
with torch.no_grad():
    ARM_aligned = model(ARM)
    cos_sim = torch.nn.functional.cosine_similarity(ARM_aligned, X86).mean().item()
    print(f"平均 Cosine 相似度: {cos_sim:.4f}")

# === 存模型 ===
torch.save(model.state_dict(), "alignment_model.pth")