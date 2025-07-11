import json
import os
from collections import defaultdict
from itertools import combinations
import pandas as pd

def extract_functions_from_json(json_file):
    """從 JSON 檔案提取函數名稱和對應的 opcodes"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        functions = {}
        
        for func_addr, func_info in data.items():
            func_name = func_info.get('function_name', f'func_{func_addr}')
            opcodes = set()
            
            if 'instructions' in func_info:
                for instruction in func_info['instructions']:
                    if 'opcode' in instruction:
                        opcodes.add(instruction['opcode'])
            
            functions[func_name] = opcodes
        
        return functions
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"❌ 錯誤檔案: {json_file}")
        return None

def jaccard_similarity(set1, set2):
    """計算兩個集合的 Jaccard 相似度"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def get_arch_from_filename(filename):
    """從檔名提取架構資訊"""
    if 'arm_32' in filename:
        return 'arm_32'
    elif 'arm_64' in filename:
        return 'arm_64'
    elif 'x86_64' in filename:
        return 'x86_64'
    elif 'mips_32' in filename:
        return 'mips_32'
    return 'unknown'

def extract_functions_from_dir(dir_path):
    """從目錄中的所有 JSON 檔案提取函數資訊"""
    all_functions = {}
    
    if not os.path.exists(dir_path):
        print(f"目錄不存在: {dir_path}")
        return None
        
    for filename in os.listdir(dir_path):
        if filename.endswith('.json'):
            filepath = os.path.join(dir_path, filename)
            functions = extract_functions_from_json(filepath)
            if functions is None:
                return None
            all_functions.update(functions)
    
    return all_functions

def remove_arch_from_name(dirname):
    """移除目錄名稱中的架構部分"""
    parts = dirname.split('_')
    arch_keywords = ['arm', 'x86', 'mips']
    
    for i, part in enumerate(parts):
        if any(arch in part for arch in arch_keywords):
            return '_'.join(parts[:i] + parts[i+2:])
    return dirname

def compare_function_similarity(results_dir):
    """比較同一程式不同架構中同名函數的相似度"""
    files_by_program = defaultdict(list)
    
    for dirname in os.listdir(results_dir):
        dir_path = os.path.join(results_dir, dirname)
        if os.path.isdir(dir_path):
            base_name = remove_arch_from_name(dirname)
            arch = get_arch_from_filename(dirname)
            files_by_program[base_name].append((dirname, arch, dir_path))
    
    results = []
    
    for program, dirs in files_by_program.items():
        if len(dirs) < 2:
            continue
            
        program_functions = {}
        
        for dirname, arch, dir_path in dirs:
            functions = extract_functions_from_dir(dir_path)
            if functions is None:
                print(f"⚠️ 跳過程式 {program}")
                break
            program_functions[arch] = functions
        else:
            # 找出所有架構共同的函數名稱
            common_functions = set(program_functions[list(program_functions.keys())[0]].keys())
            for arch_functions in program_functions.values():
                common_functions &= set(arch_functions.keys())
            
            # 比較每對架構中的同名函數
            for arch1, arch2 in combinations(program_functions.keys(), 2):
                for func_name in common_functions:
                    opcodes1 = program_functions[arch1][func_name]
                    opcodes2 = program_functions[arch2][func_name]
                    
                    if not opcodes1 and not opcodes2:
                        continue
                    
                    similarity = jaccard_similarity(opcodes1, opcodes2)
                    results.append({
                        'program': program,
                        'function': func_name,
                        'arch1': arch1,
                        'arch2': arch2,
                        'similarity': similarity
                    })
    
    return results

def print_arch_summary(results):
    """輸出四個架構的統計"""
    print("\n" + "="*40)
    print("架構組合統計")
    print("="*40)
    
    arch_pairs = defaultdict(list)
    
    for result in results:
        pair = tuple(sorted([result['arch1'], result['arch2']]))
        arch_pairs[pair].append(result['similarity'])
    
    print(f"{'架構組合':<20} {'平均相似度':<12} {'函數對數':<10}")
    print("-" * 45)
    
    for pair, similarities in sorted(arch_pairs.items()):
        avg_sim = sum(similarities) / len(similarities)
        print(f"{f'{pair[0]} vs {pair[1]}':<20} {avg_sim:.4f}       {len(similarities)}")

if __name__ == "__main__":
    results_directory = os.path.expanduser("/home/tommy/Projects/pcodeFcg/document/copied_binaries_O0_output_thunk/results")
    
    results = compare_function_similarity(results_directory)
    
    print(f"\n✅ 成功比較 {len(results)} 個函數對")
    
    print_arch_summary(results)
    

    df = pd.DataFrame(results)
    df['similarity'] = df['similarity'].round(4)
    df.to_csv('function_similarity_results_O0_thunk_function.csv', index=False)
    print(f"\n結果已儲存到 function_similarity_results.csv")