from itertools import islice
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pickle
from tqdm import tqdm
from collections import Counter
from gensim.models import Word2Vec
from preprocessing import iterate_json_files, _opcode_pat, _operand_pattern, _map_operand

DATA_DIR = Path("/home/tommy/Projects/cross-architecture/reverse/output_new/results")
TRAIN_CSV_PATH = Path("/home/tommy/Projects/pcodeFcg/dataset/csv/temp/train.csv")
OUTPUT_DIR = Path("/home/tommy/Projects/pcodeFcg/vector/contrastive/word2vec/CBOW_v2")
PICKLE_PATH = OUTPUT_DIR / "sentences_train.pkl"
BATCH_FILES = 1000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_instruction_sentence(instruction_dict):
    operation_str = instruction_dict.get("operation", "")
    if not operation_str:
        print("Warning: No operation found in instruction")
        return None
    
    command_match = _opcode_pat.search(operation_str)
    if not command_match:
        print(f"Warning: No opcode pattern found in: {operation_str}")
        return None

    command = command_match.group(1)
    sentence = [command]
    
    operands = _operand_pattern.findall(operation_str)
    if operands:
        for op_type, _ in operands: 
            mapped_operand = _map_operand(op_type)
            sentence.append(mapped_operand)
    
    return sentence

def extract_sentences_from_file(file_name_data):
    file_name, pcode_dict = file_name_data
    sentences = []

    try:
        for func_data in pcode_dict.values():
            for instruction in func_data.get("instructions", []):
                sentence = create_instruction_sentence(instruction)
                if sentence: 
                    sentences.append(sentence)
                
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

    return sentences

def corpus_generator(csv_path: Path, root_dir: Path, batch_files: int, pickle_path: Path):
    file_iter = iterate_json_files(csv_path, root_dir)
    all_sentences = []
    batch_num = 0

    while True:
        current_batch = list(islice(file_iter, batch_files))
        if not current_batch:
            break
        with Pool(cpu_count()) as pool:
            for sent_list in tqdm(pool.imap_unordered(extract_sentences_from_file, current_batch, chunksize=1),
                                  total=len(current_batch),
                                  desc=f"Batch {batch_num}"):
                all_sentences.extend(sent_list)
        batch_num += 1

    with open(pickle_path, "wb") as f:
        pickle.dump(all_sentences, f)
    return all_sentences

def analyze_tokens(sentences, output_path: Path):    
    all_tokens = []
    for sentence in sentences:
        all_tokens.extend(sentence)
    
    token_counts = Counter(all_tokens)
    unique_tokens = sorted(token_counts.keys())
    token_analysis_path = output_path / "token_analysis.txt"
    with open(token_analysis_path, "w", encoding="utf-8") as f:
        f.write(f"=== Token Analysis ===\n")
        f.write(f"Total unique tokens: {len(unique_tokens)}\n")
        f.write(f"Total token occurrences: {sum(token_counts.values())}\n\n")
        
        f.write("=== Token Frequency (sorted by frequency, descending) ===\n")
        for token, count in token_counts.most_common():
            f.write(f"{token}: {count}\n")
        
        f.write("\n=== All Unique Tokens (alphabetical order) ===\n")
        for token in unique_tokens:
            f.write(f"{token}\n")
    
    print(f"\nToken analysis saved to: {token_analysis_path}")
    print(f"Total unique tokens: {len(unique_tokens)}")
    print(f"Total token occurrences: {sum(token_counts.values())}")
    
    return token_counts

def train_word2vec(sentences, output_path: Path):
    print("\n=== 範例前五筆 sentences ===")
    for i, sent in enumerate(sentences[:5]):
        print(f"{i+1}: {sent}")

    token_counts = analyze_tokens(sentences, output_path)

    model = Word2Vec(
        sentences,
        vector_size=256,
        window=4,
        min_count=3,
        workers=cpu_count(),
        seed=42,
    )
    print("\nTraining done. Saving model…")
    model.save(str(output_path / "word2vec_x86.model"))
    print("Model size: ", len(model.wv.key_to_index))

if __name__ == "__main__":
    print("Extracting sentences and saving pickle...")
    sentences = corpus_generator(TRAIN_CSV_PATH, DATA_DIR, BATCH_FILES, PICKLE_PATH)

    print("Training Word2Vec...")
    train_word2vec(sentences, OUTPUT_DIR)