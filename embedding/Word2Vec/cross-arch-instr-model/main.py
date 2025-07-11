import numpy as np
from gensim.models import Word2Vec
from gensim.models.translation_matrix import TranslationMatrix

# 準備資料
# source_functions: 原始語言的 function (每個 function 是 opcode list)
source_functions = [
    ["LOAD", "STORE", "ADD", "INDIRECT"],
    ["PUSH", "POP", "CALL", "RET"],
    ["MOV", "JMP", "CMP", "INDIRECT"]
]

# target_functions: 目標語言的 function (對齊好的)
target_functions = [
    ["LD", "ST", "PLUS", "PTR"],
    ["PSH", "PP", "INVOKE", "RETURN"],
    ["MOVE", "JUMP", "COMPARE", "PTR"]
]

# 訓練兩個 Word2Vec 模型
source_model = Word2Vec(source_functions, vector_size=100, window=3, min_count=1, sg=1)
target_model = Word2Vec(target_functions, vector_size=100, window=3, min_count=1, sg=1)

# 建立對齊字典 (source -> target)
alignment_dict = {
    "LOAD": "LD",
    "STORE": "ST", 
    "ADD": "PLUS",
    "INDIRECT": "PTR",
    "PUSH": "PSH",
    "POP": "PP",
    "CALL": "INVOKE",
    "RET": "RETURN",
    "MOV": "MOVE",
    "JMP": "JUMP",
    "CMP": "COMPARE"
}

# 建立 translation matrix (加入 monolingual objective)
tm = TranslationMatrix(source_model.wv, target_model.wv)

# 訓練參數包含 monolingual 約束
tm.train(
    alignment_dict,
    epochs=100,
    gc=1,           # 是否使用全局一致性
    tol=1e-4,       # 收斂容忍度
    lr=0.1          # 學習率
)

# Monolingual 一致性檢查
print("Monolingual 一致性測試:")
# 檢查 source 空間的語義保持
source_similar_before = source_model.wv.most_similar("INDIRECT", topn=3)
print(f"原始 source 空間 INDIRECT 相似詞: {source_similar_before}")

# 檢查 target 空間的語義保持  
target_similar_before = target_model.wv.most_similar("PTR", topn=3)
print(f"原始 target 空間 PTR 相似詞: {target_similar_before}")

# 翻譯測試
print("\n翻譯測試:")
similar_words = tm.translate(["INDIRECT"], topn=3)
print(f"INDIRECT 的翻譯: {similar_words}")

# Joint embedding 空間測試
translated_vector = tm.translate_nn(source_model.wv["INDIRECT"])
target_similar_after = target_model.wv.most_similar([translated_vector], topn=3)
print(f"Joint 空間中的相似詞: {target_similar_after}")

# 計算翻譯品質 (cosine similarity)
if "PTR" in target_model.wv:
    target_vec = target_model.wv["PTR"]
    similarity = np.dot(translated_vector, target_vec) / (np.linalg.norm(translated_vector) * np.linalg.norm(target_vec))
    print(f"INDIRECT -> PTR 翻譯相似度: {similarity:.4f}")