
import json
from pathlib import Path
from collections import Counter

from load_data import load_jsonl
from tokenizer import tokenize

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(tokenized_texts, min_freq=1):
    """根据训练集分词结果构建词表"""

    counter = Counter()

    for tokens in tokenized_texts:
        counter.update(tokens)

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for word, freq in counter.most_common():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def text_to_ids(tokens, vocab):
    """将词语转换成数字编号"""

    return [
        vocab.get(word, vocab[UNK_TOKEN])
        for word in tokens
    ]


if __name__ == "__main__":

    # 只使用训练集构建词表
    train_path = "data/raw/TNEWS/train.json"

    print("正在读取训练集...")
    train_data = load_jsonl(train_path)

    print("正在进行中文分词...")
    tokenized_texts = [
        tokenize(item["sentence"])
        for item in train_data
    ]

    print("正在构建词表...")
    vocab = build_vocab(tokenized_texts, min_freq=1)

    # 保存词表
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "vocab.json"

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("\n===== 词表构建完成 =====")
    print(f"训练新闻数量：{len(train_data)}")
    print(f"词表大小：{len(vocab)}")
    print(f"词表保存位置：{vocab_path}")

    # 检查词表转换效果
    test_sentence = "苹果发布新款手机"
    test_tokens = tokenize(test_sentence)
    test_ids = text_to_ids(test_tokens, vocab)

    print("\n测试句子：", test_sentence)
    print("分词结果：", test_tokens)
    print("数字编号：", test_ids)
