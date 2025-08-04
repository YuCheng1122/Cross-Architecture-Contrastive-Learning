import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from model import SimSiamAdapter, simsiam_loss

class PairDataset(Dataset):
    """對比式資料集，只選正樣本進行 SimSiam 訓練"""
    def __init__(self, arm_vecs, mips_vecs, labels):
        mask = labels == 1
        self.arm = torch.from_numpy(arm_vecs[mask]).float()
        self.mips = torch.from_numpy(mips_vecs[mask]).float()

    def __len__(self):
        return self.arm.size(0)

    def __getitem__(self, idx):
        return self.arm[idx], self.mips[idx]


def train_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    count = 0
    for arm, mips in loader:
        arm = arm.to(device)
        mips = mips.to(device)
        # forward
        z1, p1 = model(arm)
        z2, p2 = model(mips)
        # SimSiam loss
        loss = 0.5 * (simsiam_loss(p1, z2.detach()) + simsiam_loss(p2, z1.detach()))
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * arm.size(0)
        count += arm.size(0)
    return total_loss / count

def evaluate_alignment(model, loader, device):
    """
    回傳平均 cosine similarity 和平均 L2 distance：
      - adapter backbone 輸出 f = model.backbone(x)
    """
    model.eval()
    sims = []
    dists = []
    with torch.no_grad():
        for arm, mips in loader:
            arm  = arm.to(device)
            mips = mips.to(device)
            # 取 adapter backbone 的輸出
            f_arm  = model.backbone(arm)   # (B, dim)
            f_mips = model.backbone(mips)  # (B, dim)
            # normalize
            f_arm_n  = F.normalize(f_arm, dim=1)
            f_mips_n = F.normalize(f_mips, dim=1)
            # cosine similarity
            sim = (f_arm_n * f_mips_n).sum(dim=1)    # (B,)
            sims.append(sim.cpu())
            # L2 distance
            dist = torch.norm(f_arm - f_mips, p=2, dim=1)
            dists.append(dist.cpu())
    sims   = torch.cat(sims)
    dists  = torch.cat(dists)
    return sims.mean().item(), dists.mean().item()



def main():
    # 1) 載入資料
    arm_vecs  = np.load("/home/tommy/Projects/pcodeFcg/vector/contrastive/resnet/mips/X_mips.npy")
    x86_vecs = np.load("/home/tommy/Projects/pcodeFcg/vector/contrastive/resnet/mips/X_x86.npy")
    labels    = np.load("/home/tommy/Projects/pcodeFcg/vector/contrastive/resnet/mips/y.npy")

    # 2) 建立 DataLoader，只保留 label=1
    dataset = PairDataset(arm_vecs, x86_vecs, labels)
    loader  = DataLoader(dataset, batch_size=128, shuffle=True)

    # 3) 設定裝置、模型、優化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SimSiamAdapter(dim=256, proj_hidden=2048, pred_hidden=512, num_blocks=2).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 4) 訓練
    epochs = 50
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optim, device)
        print(f"Epoch {epoch:02d}: SimSiam Loss = {loss:.4f}")

    # 5) 儲存模型
    torch.save(model.state_dict(), "/home/tommy/Projects/pcodeFcg/vector/contrastive/resnet/mips/simsiam_adapter.pth")
    print("✅ Model saved to simsiam_adapter.pth")

if __name__ == "__main__":
    main()
