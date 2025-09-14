import os
import torch
import pickle
import networkx as nx
from tqdm import tqdm


# === AlignmentModel 定義 ===
class AlignmentModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


# === 路徑設定 ===
model_path = "/home/tommy/Projects/pcodeFcg/embedding/Procrustes/alignment_model.pth"
data_path = "/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/arm_cbow_v2/train"
output_path = "/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/arm_cbow_v2/train_transferred_align"
os.makedirs(output_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 載入 Alignment 模型 ===
dim = 256
model = AlignmentModel(dim).to(device)
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

    # Alignment
    with torch.inference_mode():
        new_vecs = model(vectors).cpu().numpy()

    # 更新節點向量
    for (node, data), new_vec in zip(G.nodes(data=True), new_vecs):
        data["vector"] = new_vec

    # 輸出存檔
    with open(dst_path, "wb") as fp:
        pickle.dump(G, fp)

print("✅ Alignment 完成，結果已存到:", output_path)
