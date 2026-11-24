
"""
TNEWS 数据集读取模块
支持读取有标签和无标签的数据。
"""

import json
from pathlib import Path


def load_jsonl(file_path):
    """逐行读取 JSONL 文件。"""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件：{file_path}")

    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"第 {line_number} 行格式错误：{e}"
                )

            if not isinstance(item, dict):
                raise ValueError(
                    f"第 {line_number} 行不是 JSON 对象"
                )

            if "sentence" not in item:
                raise ValueError(
                    f"第 {line_number} 行缺少 sentence 字段"
                )

            data.append(item)

    return data


def check_data(data, dataset_name="数据集"):
    """检查数据集基本情况。"""

    print(f"\n===== {dataset_name} =====")
    print(f"数据总量：{len(data)}")

    if not data:
        print("数据集为空")
        return

    print("第一条数据：")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))

    labeled_count = sum(
        1 for item in data if "label" in item
    )

    print(f"有标签的数据：{labeled_count}")
    print(f"无标签的数据：{len(data) - labeled_count}")



if __name__ == "__main__":

    # TNEWS 数据集路径
    data_dir = "data/raw/TNEWS"

    # 读取训练集
    train_data = load_jsonl(f"{data_dir}/train.json")
    check_data(train_data, "训练集")

    # 读取验证集
    dev_data = load_jsonl(f"{data_dir}/dev.json")
    check_data(dev_data, "验证集")

    # 读取测试集
    test_data = load_jsonl(f"{data_dir}/test.json")
    check_data(test_data, "测试集")
