import json
from os import name
import re
from pathlib import Path
from typing import List, Dict, Tuple, Generator
from unicodedata import category
import pandas as pd
from tqdm import tqdm
#Regex pattern preprocessing
#1)  opcode_pattern: Extract P-Code
#2)  opcode_pattern: Extract Calculation

__all__ = ["iterate_json_files", "opcode_pat", "operand_pattern", "_map_operand"]

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