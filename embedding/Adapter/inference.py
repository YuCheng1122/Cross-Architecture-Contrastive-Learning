import os
import pickle
import torch
from tqdm import tqdm

from model import LSTMAdapter, MLPAdapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adapter = MLPAdapter(input_dim=256).to(device)
adapter.load_state_dict(torch.load("ARM_ResNet_adapter_10.pth", map_location=device))
adapter.eval()

data_path = "/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/arm_cbow_v2/test_arm"
output_path = "/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/arm_cbow_v2/test_arm_transferred_ResNet10"
os.makedirs(output_path, exist_ok=True)

gpickle_files = []
for root, _, files in os.walk(data_path):
    rel_folder = os.path.relpath(root, data_path)
    target_folder = os.path.join(output_path, rel_folder)
    os.makedirs(target_folder, exist_ok=True)

    for fname in files:
        if fname.endswith(".gpickle"):
            gpickle_files.append((os.path.join(root, fname),
                                  os.path.join(target_folder, fname)))

for src_path, dst_path in tqdm(gpickle_files, desc="Transferring embeddings"):
    with open(src_path, "rb") as fp:
        G = pickle.load(fp)
    vectors = [torch.tensor(data["vector"], dtype=torch.float32) for _, data in G.nodes(data=True)]
    if not vectors:  
        with open(dst_path, "wb") as fp:
            pickle.dump(G, fp)
        continue

    vectors = torch.stack(vectors).to(device)

    with torch.inference_mode():
        new_vecs = adapter(vectors).cpu().numpy()

    for (node, data), new_vec in zip(G.nodes(data=True), new_vecs):
        data["vector"] = new_vec

    with open(dst_path, "wb") as fp:
        pickle.dump(G, fp)
