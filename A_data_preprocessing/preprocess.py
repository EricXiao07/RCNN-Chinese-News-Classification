
import json
from pathlib import Path

from load_data import load_jsonl
from tokenizer import tokenize

# =========================
# 1. 基本配置
# =========================

DATA_DIR = Path("data/raw/TNEWS")
OUTPUT_DIR = Path("data/processed")

VOCAB_PATH = OUTPUT_DIR / "vocab.json"
LABEL2ID_PATH = OUTPUT_DIR / "label2id.json"

# 当前先使用 50 作为初始长度，后续可以根据文本长度分布调整
MAX_LEN = 50

PAD_ID = 0
UNK_ID = 1


# =========================
# 2. 加载词表和标签映射
# =========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 3. 填充与截断
# =========================

def pad_or_truncate(token_ids, max_len=MAX_LEN):
    if len(token_ids) > max_len:
        return token_ids[:max_len]

    if len(token_ids) < max_len:
        token_ids = token_ids + [PAD_ID] * (max_len - len(token_ids))

    return token_ids


# =========================
# 4. 文本转换为模型输入
# =========================

def text_to_ids(text, vocab):
    tokens = tokenize(text)
    token_ids = [vocab.get(word, UNK_ID) for word in tokens]

    original_length = len(token_ids)
    input_ids = pad_or_truncate(token_ids)

    return input_ids, original_length


# =========================
# 5. 处理一个数据集
# =========================

def process_dataset(data, vocab, label2id, dataset_name):
    processed_data = []

    for item in data:
        sentence = item["sentence"]

        input_ids, original_length = text_to_ids(sentence, vocab)

        result = {
            "input_ids": input_ids,
            "sentence": sentence,
            "original_length": original_length
        }

        # 训练集和验证集有标签
        if "label" in item:
            original_label = str(item["label"])

            if original_label not in label2id:
                raise ValueError(
                    f"发现未知标签：{original_label}"
                )

            result["label"] = label2id[original_label]
            result["original_label"] = original_label
            result["label_desc"] = item.get("label_desc", "")

        # 测试集没有真实标签，保留原始 id
        if "id" in item:
            result["id"] = item["id"]

        processed_data.append(result)

    return processed_data


# =========================
# 6. 保存为 JSONL 文件
# =========================

def save_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# =========================
# 7. 主程序
# =========================

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== RCNN 数据预处理开始 =====")

    # 加载词表和标签映射
    vocab = load_json(VOCAB_PATH)
    label2id = load_json(LABEL2ID_PATH)

    print("词表大小：", len(vocab))
    print("类别数量：", len(label2id))

    # 读取原始数据
    train_data = load_jsonl(DATA_DIR / "train.json")
    dev_data = load_jsonl(DATA_DIR / "dev.json")
    test_data = load_jsonl(DATA_DIR / "test.json")

    print("\n原始数据读取完成")
    print("训练集：", len(train_data))
    print("验证集：", len(dev_data))
    print("测试集：", len(test_data))

    # 分别处理三个数据集
    datasets = {
        "train": train_data,
        "dev": dev_data,
        "test": test_data
    }

    for name, data in datasets.items():

        print(f"\n正在处理 {name} 数据集……")

        processed_data = process_dataset(
            data,
            vocab,
            label2id,
            name
        )

        output_path = OUTPUT_DIR / f"{name}.jsonl"
        save_jsonl(processed_data, output_path)

        print(f"{name} 处理完成")
        print("处理后数量：", len(processed_data))
        print("保存位置：", output_path)

    print("\n===== 数据预处理全部完成 =====")
