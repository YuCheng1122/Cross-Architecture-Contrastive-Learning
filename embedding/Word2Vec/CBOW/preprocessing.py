import json
from os import name
import re
from pathlib import Path
from typing import List, Dict, Tuple, Generator
from unicodedata import category
import pandas as pd
from tqdm import tqdm

__all__ = ['read_csv', 'load_json', 'Pcode_to_sentence']

#Regex pattern preprocessing
#1)  opcode_pattern: Extract P-Code
#2)  opcode_pattern: Extract Calculation

_opcode_pat = re.compile(r"(?:\)\s+|---\s+)([A-Z_]+)")
_operand_pattern = re.compile(r"\(([^ ,]+)\s*,\s*[^,]*,\s*([0-9]+)\)")

def read_csv(csv_file_path: str | Path) -> List[List[str]]:
    df = pd.read_csv(csv_file_path)
    file_names = df['file_name'].tolist()
    return file_names

def iterate_json_files(csv_file_path: str | Path, root_dir: str | Path) -> Generator[Tuple[Path, Dict], None, None]:
    root_path = Path(root_dir)
    file_names = read_csv(csv_file_path)
    for file_name in tqdm(file_names, desc="Processing JSON files"):
        json_path = root_path / file_name / f"{file_name}.json"
        
        if json_path.exists():
            try:
                with json_path.open(encoding="utf-8") as fp:
                    yield json_path, json.load(fp)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {json_path}")
        else:
            print(f"File not found: {json_path}")

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

def _tokenize_line(line:str) -> List[str]:
    command_match = _opcode_pat.search(line)
    if not command_match:
        return []

    command = command_match.group(1)
    operands = _operand_pattern.findall(line)
    types = [_map_operand(op)for op, _ in operands]

    if len(types) > 5:
        types = types[:4] + ["MIX"]

    token = "-".join([command] + types)

    if len(types) > 5 or token.count("-") > 5:
        print("⚠️ 長度過長:", token)
        assert False, f"Token too long: {token}"
    return [token]
