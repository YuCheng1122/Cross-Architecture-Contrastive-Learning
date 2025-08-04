import os
import torch
import networkx as nx
import pickle

# 1. Adapter 類定義
from model import Adapter  

# 2. 讀 checkpoint 並取 state_dict
model_path = '/home/tommy/Projects/pcodeFcg/vector/contrastive/resnet/mips/simsiam_adapter.pth'
ckpt = torch.load(model_path, map_location='cpu')
state_dict = ckpt.get('state_dict', ckpt)

# 3. 過濾出 adapter 權重（去掉 'backbone.' 前綴）
adapter_state = {
    k[len('backbone.'):]: v
    for k, v in state_dict.items()
    if k.startswith('backbone.adapter.')
}

print('Loaded state_dict keys:', list(adapter_state.keys()))

# 4. 建立 Adapter 實例並載入權重
adapter = Adapter(dim=256, num_blocks=2)
missing, unexpected = adapter.load_state_dict(adapter_state, strict=False)
print('Missing keys:', missing)
print('Unexpected keys:', unexpected)
adapter.eval()

# 5. 指定輸入/輸出資料夾
src_dir = '/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/mips/test'
dst_dir = '/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN//mips/test_transfer'
os.makedirs(dst_dir, exist_ok=True)

for root, _, files in os.walk(src_dir):
    rel_folder = os.path.relpath(root, src_dir)
    target_folder = os.path.join(dst_dir, rel_folder)
    os.makedirs(target_folder, exist_ok=True)

    for fname in files:
        if not fname.endswith('.gpickle'):
            continue

        src_path = os.path.join(root, fname)
        dst_path = os.path.join(target_folder, fname)

        # 4. 用 pickle.load 開啟
        with open(src_path, 'rb') as fp:
            G = pickle.load(fp)

        # 5. 轉換每個 node 的 vector
        for node, data in G.nodes(data=True):
            vec = torch.tensor(data['vector'], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                new_vec = adapter(vec).squeeze(0).numpy()
            G.nodes[node]['vector'] = new_vec

        # 6. 用 pickle.dump 寫回
        with open(dst_path, 'wb') as fp:
            pickle.dump(G, fp)

        print(f'Transferred {os.path.join(rel_folder, fname)}')