import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple
from functools import partial
import multiprocessing as mp

import networkx as nx
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec

_opcode_pat = re.compile(r"(?:\)\s+|---\s+)([A-Z_]+)")
_operand_pattern = re.compile(r"\(([^ ,]+)\s*,\s*[^,]*,\s*([0-9]+)\)")

def read_csv(csv_file_path: str | Path) -> List[str]:
    df = pd.read_csv(csv_file_path)
    return df['file_name'].tolist()

def _map_operand(op_type: str) -> str:
    op_type_l = op_type.lower()
    if op_type_l == 'register':
        return "REG"
    if op_type_l == 'ram':
        return "MEM"
    if op_type_l in {'const', 'constant'}:
        return "CONST"
    if op_type_l == 'unique':
        return "UNIQUE"
    if op_type_l == 'stack':
        return "STACK"
    return "UNK"


def clean_data(json_data, G_raw: nx.DiGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in G_raw.nodes():
        addr = str(node)
        func = json_data.get(addr)
        if not func:
            continue
        instructions = func.get("instructions", [])
        pcode_list = []
        for instr in instructions:
            if isinstance(instr, dict):
                operation = instr.get("operation")
                if isinstance(operation, str):
                    pcode_list.append(operation)
        if pcode_list:
            G.add_node(addr, operation=pcode_list)
    for src, dst in G_raw.edges():
        src, dst = str(src), str(dst)
        if G.has_node(src) and G.has_node(dst):
            G.add_edge(src, dst)
    return G

def parse_tokens_from_operation(operation_str: str) -> list[str]:
    if not operation_str:
        return []
    command_match = _opcode_pat.search(operation_str)
    if not command_match:
        return []
    command = command_match.group(1)
    tokens = [command]

    operands = _operand_pattern.findall(operation_str)
    if operands:
        for op_type, _ in operands:
            mapped_operand = _map_operand(op_type)
            tokens.append(mapped_operand)
    return tokens

def vectorize_graph_nodes(G: nx.Graph, model: Word2Vec):
    zero_vec = model.wv.vectors[0] * 0
    for _, data in G.nodes(data=True):
        operation_lines = data.get("operation", [])
        tokens = [tok for line in operation_lines for tok in parse_tokens_from_operation(line)]
        vectors = [model.wv[t] for t in tokens if t in model.wv]
        data["vector"] = sum(vectors) / len(vectors) if vectors else zero_vec
        data.pop("operation", None)

def process_single_file(file_info: Tuple[Path, Path, str], output_base_path: Path, model: Word2Vec) -> str:
    json_path, dot_path, file_name = file_info
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        G_raw = nx.drawing.nx_pydot.read_dot(dot_path)
        G = clean_data(json_data, G_raw)
        vectorize_graph_nodes(G, model)

        prefix = file_name[:2]
        output_dir = output_base_path / prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{file_name}.gpickle"
        with open(out_path, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        return f"Processed {file_name}"
    except Exception as e:
        return f"Error {file_name}: {str(e)}"


def process_all(csv_file_path: Path, root_dir: Path, out_dir: Path, w2v_model_path: Path, num_processes=None):
    file_names = read_csv(csv_file_path)
    file_info_list = []
    for file_name in tqdm(file_names, desc="Collecting file paths"):
        json_path = root_dir / file_name / f"{file_name}.json"
        dot_path = root_dir / file_name / f"{file_name}.dot"
        if json_path.exists() and dot_path.exists():
            file_info_list.append((json_path, dot_path, file_name))
        else:
            print(f"[Missing] {file_name}: {'json' if not json_path.exists() else ''} {'dot' if not dot_path.exists() else ''}")

    model = Word2Vec.load(str(w2v_model_path))
    if num_processes is None:
        num_processes = mp.cpu_count()

    process_func = partial(process_single_file, output_base_path=out_dir, model=model)

    print(f"Processing {len(file_info_list)} files using {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_func, file_info_list), total=len(file_info_list), desc="Processing"))

    # 總結
    success = [r for r in results if r.startswith("Processed")]
    fail = [r for r in results if r.startswith("Error")]

if __name__ == "__main__":
    CSV_FILE = Path("/home/tommy/Projects/pcodeFcg/dataset/csv/temp/test_mips.csv")
    ROOT_DIR = Path("/home/tommy/Projects/cross-architecture/reverse/output_new/results")
    OUT_DIR = Path("/home/tommy/Projects/pcodeFcg/vector/contrastive/GNN/mips_cbow_v2/test_mips")
    W2V_MODEL = Path("/home/tommy/Projects/pcodeFcg/vector/contrastive/word2vec/CBOW_v2/word2vec_x86.model")

    process_all(CSV_FILE, ROOT_DIR, OUT_DIR, W2V_MODEL)
