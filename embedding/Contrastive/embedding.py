import os
import torch
import pickle
import networkx as nx
from tqdm import tqdm

from model import SimSiam, MLP

model_path = "/home/tommy/Projects/pcodeFcg/embedding/Contrastive/best_model_Os.pth"
data_path = "/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/arm_cbow_v2/test_arm"
output_path = "/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/arm_cbow_v2/test_arm_transferred_Os"
os.makedirs(output_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimSiam(base_encoder=MLP, dim=256, pred_dim=128).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

gpickle_files = []
for root, _, files in os.walk(data_path):
    rel_folder = os.path.relpath(root, data_path)
    target_folder = os.path.join(output_path, rel_folder)
    os.makedirs(target_folder, exist_ok=True)

    for fname in files:
        if fname.endswith(".gpickle"):
            gpickle_files.append((os.path.join(root, fname),
                                  os.path.join(target_folder, fname)))
for src_path, dst_path in tqdm(gpickle_files, desc="Vectorizing graphs"):
    with open(src_path, "rb") as fp:
        G = pickle.load(fp)

    vectors = [torch.tensor(data["vector"], dtype=torch.float32) for _, data in G.nodes(data=True)]
    vectors = torch.stack(vectors).to(device)

    with torch.inference_mode():
        f = model.encoder(vectors)
        z = model.projector(f)
        new_vecs = z.cpu().numpy()

    for (node, data), new_vec in zip(G.nodes(data=True), new_vecs):
        data["vector"] = new_vec

    with open(dst_path, "wb") as fp:
        pickle.dump(G, fp)
