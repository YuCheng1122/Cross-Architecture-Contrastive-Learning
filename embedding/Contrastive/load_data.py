import os
import pickle
import numpy as np
from gensim.models import Word2Vec
from collections import Counter

# 路徑設定
pkl_path    = "/home/tommy/Projects/pcodeFcg/dataset/alignment/train_mips.pickle"
model_path  = "/home/tommy/Projects/pcodeFcg/vector/contrastive/word2vec/CBOW/word2vec_20250509_train_450.model"
output_dir  = "/home/tommy/Projects/pcodeFcg/vector/contrastive/resnet/mips"
os.makedirs(output_dir, exist_ok=True)

# 1. 載入資料
with open(pkl_path, "rb") as f:
    samples = pickle.load(f)

w2v = Word2Vec.load(model_path)

# 2. 紀錄 token 統計
unknown_token_counter = Counter()
total_token_counter = Counter()

def pcode_to_vec(pcode_str, model):
    tokens = pcode_str.split()
    vecs = []
    for t in tokens:
        total_token_counter[t] += 1
        if t in model.wv:
            vecs.append(model.wv[t])
        else:
            unknown_token_counter[t] += 1
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)

# 3. 轉向量
X_x86 = []
X_arm = []
y = []

for p_x, p_a, label in samples:
    vec_x = pcode_to_vec(p_x, w2v)
    vec_a = pcode_to_vec(p_a, w2v)
    X_x86.append(vec_x)
    X_arm.append(vec_a)
    y.append(label)

X_x86 = np.stack(X_x86, axis=0)
X_arm = np.stack(X_arm, axis=0)
y = np.array(y, dtype=np.int64)

# 4. 儲存向量
np.save(os.path.join(output_dir, "X_x86.npy"), X_x86)
np.save(os.path.join(output_dir, "X_mips.npy"), X_arm)
np.save(os.path.join(output_dir, "y.npy"), y)

# 5. 統計與輸出
unknown_tokens = set(unknown_token_counter)
all_tokens = set(total_token_counter)

unique_covered = len(all_tokens - unknown_tokens)
unique_total = len(all_tokens)
unique_coverage = unique_covered / unique_total * 100

total_covered = sum(total_token_counter.values()) - sum(unknown_token_counter.values())
total_tokens = sum(total_token_counter.values())
total_coverage = total_covered / total_tokens * 100

print(f"\nCoverage 統計：")
print(f"Unique tokens     = {unique_total}")
print(f"Unknown tokens    = {len(unknown_tokens)}")
print(f"Unique coverage = {unique_coverage:.2f}%")

print(f"\nTotal tokens      = {total_tokens}")
print(f"Unknown total     = {sum(unknown_token_counter.values())}")
print(f"Total coverage  = {total_coverage:.2f}%")

# 輸出 top missing tokens
top_unknowns = unknown_token_counter.most_common(20)
for tok, count in top_unknowns:
    print(f"{tok:<30} {count}")

print("\n向量轉換與統計完成，已儲存到：", output_dir)
