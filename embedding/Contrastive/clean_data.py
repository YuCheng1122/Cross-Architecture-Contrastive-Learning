import os
import json
import csv
import pickle
from preprocessing import _tokenize_line

def extract_alignment_and_generate_data():
    folder_path = "/home/tommy/Projects/pcodeFcg/document/copied_binaries_Os_output/results"
    output_path = "/home/tommy/Projects/pcodeFcg/dataset/alignment"
    archs = ["mips_32", "arm_32", "x86_64"]

    # 1. 讀取 JSON 並依 (file::function) 存每個架構的 tokenized operation
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
                for func_addr, func in data.items():
                    fn = func.get("function_name", "").strip()
                    instrs = func.get("instructions", [])
                    if not fn or not instrs:
                        continue

                    tokens = []
                    for i in instrs:
                        op = i.get("operation", "").strip()
                        if not op:
                            continue
                        toks = _tokenize_line(op)
                        tokens.extend(toks)

                    if not tokens:
                        continue

                    tokenized_line = " ".join(tokens)
                    key = f"{file_base}::{fn}"
                    arch_data[arch][key] = (file_base, fn, tokenized_line)
            except Exception as e:
                print(f"{file_base} 讀取失敗，跳過: {e}")

    # 2. 取三個架構都有的 key
    common_keys = set(arch_data["mips_32"]) & set(arch_data["arm_32"]) & set(arch_data["x86_64"])

    # 3. 過濾掉 operation 完全相同的組合
    seen_operations = set()
    filtered_keys = []
    for key in sorted(common_keys):
        ops = tuple(arch_data[arch][key][2] for arch in archs)
        if ops in seen_operations:
            continue
        seen_operations.add(ops)
        filtered_keys.append(key)

    print(f"共取得 {len(filtered_keys)} 組不重複 tokenized operation 對齊資料")

    os.makedirs(output_path, exist_ok=True)

    # 4. 輸出每個架構的對應 CSV
    for arch in archs:
        csv_path = os.path.join(output_path, f"{arch}_alignment.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "function_name", "operation", "label"])
            for key in filtered_keys:
                file_base, fn, op = arch_data[arch][key]
                writer.writerow([file_base, fn, op, 1])
        print(f"已輸出 {csv_path}")

    # 5. 輸出 x86_64 和 arm_32 的配對 pickle
    samples = []
    for key in filtered_keys:
        _, _, x86_op = arch_data["x86_64"][key]
        _, _, mips_op = arch_data["mips_32"][key]
        samples.append((x86_op, mips_op, 1))

    pk_file = os.path.join(output_path, "train_mips.pickle")
    with open(pk_file, "wb") as f:
        pickle.dump(samples, f)
    print(f"已生成 {pk_file}，共 {len(samples)} 筆樣本")

if __name__ == "__main__":
    extract_alignment_and_generate_data()
