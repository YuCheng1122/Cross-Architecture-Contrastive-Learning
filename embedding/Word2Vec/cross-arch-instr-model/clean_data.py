
import json
import os
from pathlib import Path

def extract_opcodes_by_arch(folder_path):
   folder_path = Path(folder_path)
   
   # 儲存結果的字典，按架構分類
   arch_data = {}
   
   # 遍歷資料夾中的所有 JSON 檔案
   for json_file in folder_path.glob("*.json"):
       # 從檔名解析架構資訊
       filename = json_file.stem
       parts = filename.split('_')
       
       # 判斷架構
       if 'x86' in filename.lower():
           arch = 'x86'
       elif 'arm' in filename.lower():
           arch = 'arm32'
       else:
           continue
           
       if arch not in arch_data:
           arch_data[arch] = {}
           
       # 讀取 JSON 檔案
       with open(json_file, 'r', encoding='utf-8') as f:
           data = json.load(f)
           
       # 提取每個函數的 opcode
       for addr, func_info in data.items():
           func_name = func_info.get('function_name', '')
           if not func_name:
               continue
               
           # 提取所有 opcode
           opcodes = []
           for instruction in func_info.get('instructions', []):
               opcode = instruction.get('opcode', '')
               if opcode:
                   opcodes.append(opcode)
           
           if func_name not in arch_data[arch]:
               arch_data[arch][func_name] = []
           arch_data[arch][func_name].extend(opcodes)
   
   # 輸出到檔案
   for arch, functions in arch_data.items():
       output_file = f"{arch}_opcodes.txt"
       with open(output_file, 'w', encoding='utf-8') as f:
           for func_name, opcodes in functions.items():
               opcode_str = ' '.join(opcodes)
               f.write(f"{func_name}: {opcode_str}\n")
       
       print(f"已生成 {output_file}")

# 使用方式
folder_path = "/home/tommy/Projects/pcodeFcg/document/copied_binaries_Os_output/results"
extract_opcodes_by_arch(folder_path)