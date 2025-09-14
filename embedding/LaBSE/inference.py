import os
import torch
import pickle
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

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

class DualEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, out_dim=256, margin=0.3, scale=10):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim, out_dim)
        self.margin = margin
        self.scale = scale
    def forward(self, src_vecs, tgt_vecs):
        src_emb = self.encoder(src_vecs)
        tgt_emb = self.encoder(tgt_vecs)
        return src_emb, tgt_emb


# === 檔案路徑 ===
model_path = "dual_encoder_alignment.pth"
data_path = "/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/arm_cbow_v2/train"
output_path = "/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/arm_cbow_v2/train_transferred_align_labse"
os.makedirs(output_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 載入 Dual Encoder 模型 ===
dim = 256
model = DualEncoder(input_dim=dim).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# === 處理所有 gpickle 檔案 ===
gpickle_files = []
for root, _, files in os.walk(data_path):
    rel_folder = os.path.relpath(root, data_path)
    target_folder = os.path.join(output_path, rel_folder)
    os.makedirs(target_folder, exist_ok=True)

    for fname in files:
        if fname.endswith(".gpickle"):
            gpickle_files.append((os.path.join(root, fname),
                                  os.path.join(target_folder, fname)))

for src_path, dst_path in tqdm(gpickle_files, desc="Aligning graphs"):
    with open(src_path, "rb") as fp:
        G = pickle.load(fp)

    # 取出 node 向量
    vectors = [torch.tensor(data["vector"], dtype=torch.float32) for _, data in G.nodes(data=True)]
    if len(vectors) == 0:
        continue
    vectors = torch.stack(vectors).to(device)

    # Alignment (只用 encoder 即可)
    with torch.inference_mode():
        new_vecs = model.encoder(vectors).cpu().numpy()

    # 更新節點向量
    for (node, data), new_vec in zip(G.nodes(data=True), new_vecs):
        data["vector"] = new_vec

    # 輸出存檔
    with open(dst_path, "wb") as fp:
        pickle.dump(G, fp)

print("✅ Alignment 完成，結果已存到:", output_path)
