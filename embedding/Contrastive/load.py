import os
import json
import pickle
import random
import csv
import numpy as np
from gensim.models import Word2Vec
from collections import Counter
from preprocessing import _opcode_pat, _operand_pattern, _map_operand


def _tokens_from_operation(operation_str: str) -> list[str]:
    if not operation_str:
        return []
    m = _opcode_pat.search(operation_str)
    if not m:
        return []
    opcode = m.group(1)
    tokens = [opcode]
    operands = _operand_pattern.findall(operation_str)
    for op_type, _ in operands:
        tokens.append(_map_operand(op_type))
    return tokens


def load_data_from_folder(folder_path, archs):
    arch_data = {arch: {} for arch in archs}
    for root, _, files in os.walk(folder_path):
        subfolder = os.path.basename(root)
        arch = next((a for a in archs if a in subfolder), None)
        if not arch:
            continue

        file_base = subfolder.split("_")[-1]
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for _, func in data.items():
                    fn = func.get("function_name", "").strip()
                    instrs = func.get("instructions", [])
                    if not fn or not instrs:
                        continue

                    flat_tokens: list[str] = []
                    for ins in instrs:
                        op = ins.get("operation", "").strip()
                        if not op:
                            continue
                        sent = _tokens_from_operation(op)
                        if sent:
                            flat_tokens.extend(sent)

                    if not flat_tokens:
                        continue

                    tokenized_line = " ".join(flat_tokens)
                    key = f"{file_base}::{fn}"
                    arch_data[arch][key] = (file_base, fn, tokenized_line)

            except Exception as e:
                print(f"{file_base} 讀取失敗，跳過: {e}")
    return arch_data


def pcode_to_vec(pcode_str, model, counter_total, counter_unknown):
    tokens = pcode_str.split()
    vecs = []
    for t in tokens:
        counter_total[t] += 1
        if t in model.wv:
            vecs.append(model.wv[t])
        else:
            counter_unknown[t] += 1
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)


def extract_and_vectorize():
    # ===== 來源資料夾可以自由增加 =====
    input_folders = [
        "/home/tommy/Projects/pcodeFcg/document/copied_binaries_O0_output/results",
        "/home/tommy/Projects/pcodeFcg/document/copied_binaries_Os_output/results",
    ]

    output_path = "/home/tommy/Projects/pcodeFcg/dataset/alignment_vector"
    model_path  = "/home/tommy/Projects/pcodeFcg/vector/contrastive/word2vec/CBOW_v2/word2vec_x86.model"
    archs = ["mips_32", "arm_32", "x86_64"]

    w2v = Word2Vec.load(model_path)

    # 每個來源資料夾都跑一次
    arch_datasets = []
    for folder in input_folders:
        arch_datasets.append(load_data_from_folder(folder, archs))

    # 找所有來源的交集
    common_keys = None
    for arch_data in arch_datasets:
        if common_keys is None:
            common_keys = set(arch_data["mips_32"]) & set(arch_data["arm_32"]) & set(arch_data["x86_64"])
        else:
            common_keys &= set(arch_data["mips_32"]) & set(arch_data["arm_32"]) & set(arch_data["x86_64"])

    common_keys = list(common_keys)
    random.shuffle(common_keys)

    # 平均切給不同來源
    n = len(input_folders)
    chunk_size = len(common_keys) // n
    key_groups = [common_keys[i*chunk_size:(i+1)*chunk_size] for i in range(n)]

    # Counter
    total_token_counter = Counter()
    unknown_token_counter = Counter()

    samples = []
    for arch_data, keys in zip(arch_datasets, key_groups):
        for key in keys:
            _, _, x86_op = arch_data["x86_64"][key]
            _, _, arm_op = arch_data["arm_32"][key]
            vec_x = pcode_to_vec(x86_op, w2v, total_token_counter, unknown_token_counter)
            vec_a = pcode_to_vec(arm_op, w2v, total_token_counter, unknown_token_counter)
            samples.append((x86_op, arm_op, vec_x, vec_a, 1))

    os.makedirs(output_path, exist_ok=True)

    pk_file = os.path.join(output_path, "train_arm_vector_mix.pickle")
    with open(pk_file, "wb") as f:
        pickle.dump([(vec_x, vec_a, label) for _, _, vec_x, vec_a, label in samples], f)

    csv_file = os.path.join(output_path, "train_arm_op_mix.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x86_op", "arm_op", "label"])
        for x86_op, arm_op, _, _, label in samples:
            writer.writerow([x86_op, arm_op, label])

    # Coverage 統計
    unknown_tokens = set(unknown_token_counter)
    all_tokens = set(total_token_counter)
    unique_coverage = (len(all_tokens - unknown_tokens) / len(all_tokens)) * 100 if all_tokens else 0
    total_coverage = ((sum(total_token_counter.values()) - sum(unknown_token_counter.values())) /
                      sum(total_token_counter.values()) * 100) if total_token_counter else 0

    print(f"已生成 {pk_file} & {csv_file}，共 {len(samples)} 筆樣本")
    print(f"Unique coverage = {unique_coverage:.2f}% , Total coverage = {total_coverage:.2f}%")


if __name__ == "__main__":
    extract_and_vectorize()
