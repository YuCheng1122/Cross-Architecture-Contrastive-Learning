from pathlib import Path
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Generator, Sequence
import pandas as pd
import networkx as nx

def read_csv(csv_file_path: str | Path) -> List[List[str]]:
    df = pd.read_csv(csv_file_path)
    file_names = df['file_name'].tolist()
    return file_names

def iterate_Gpickle(
    csv_file_path: str | Path,
    root_dir: str | Path
) -> Generator[Tuple[Path, nx.DiGraph, Dict[str, Sequence[str]]], None, None]:
    root_path = Path(root_dir)
    file_names = read_csv(csv_file_path)

    for file_name in tqdm(file_names, desc="Processing Gpickle files"):
        prefix = file_name[:2]
        path = root_path / prefix / f"{file_name}.gpickle"
        
        if path.exists():
            try:
                with open(path, "rb") as fp:
                    G = pickle.load(fp)
                pcode_map = nx.get_node_attributes(G, "pcode")
                yield path, G, pcode_map
            except Exception as e:
                tqdm.write(f"[Error] Load Gpickle Failed {path}: {e}")
        else:
            tqdm.write(f"[Warning] File Not Found: {file_name}.gpickle")