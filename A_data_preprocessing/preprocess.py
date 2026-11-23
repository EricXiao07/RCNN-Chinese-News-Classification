
"""
TNEWS 数据预处理主程序

主要工作：
1. 读取训练集、验证集和测试集
2. 对新闻标题进行中文分词
3. 仅使用训练集构建词表
4. 将文本转换为数字 ID
5. 对文本进行截断和填充
6. 保存处理后的数据
"""

import json
from pathlib import Path

from tokenizer import tokenize
from build_vocab import build_vocab, text_to_ids


PAD_ID = 0
UNK_ID = 1
MAX_LEN = 50


def pad_or_truncate(ids, max_len=MAX_LEN):
    """将文本序列截断或填充到指定长度。"""
    ids = ids[:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


def process_records(records, vocab, label_mapping):
    """将原始记录转换为模型输入格式。"""
    processed = []

    for item in records:
        sentence = item.get("sentence", "")
        label = item.get("label")

        tokens = tokenize(sentence)
        input_ids = text_to_ids(tokens, vocab)
        input_ids = pad_or_truncate(input_ids)

        result = {
            "input_ids": input_ids
        }

        if label is not None and str(label) in label_mapping:
            result["label"] = label_mapping[str(label)]

        processed.append(result)

    return processed


def save_json(data, output_path):
    """保存 JSON 文件。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("TNEWS 数据预处理主程序")
    print("当前为预处理流程框架，待数据读取模块和数据集路径确定后运行。")
