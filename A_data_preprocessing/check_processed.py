
import json
from pathlib import Path

DATA_DIR = Path("data/processed")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    for name in ["train", "dev", "test"]:
        data = read_jsonl(DATA_DIR / f"{name}.jsonl")

        print(f"\n===== {name} 数据集检查 =====")
        print("数据数量：", len(data))

        # 检查输入序列长度
        lengths = [len(item["input_ids"]) for item in data]
        print("序列长度是否全部为50：", all(x == 50 for x in lengths))

        # 检查标签
        if name != "test":
            labels = [item["label"] for item in data]
            print("标签最小值：", min(labels))
            print("标签最大值：", max(labels))
            print("标签数量：", len(set(labels)))
        else:
            print("测试集是否保留 id：", all("id" in item for item in data))
            print("测试集是否错误添加 label：", any("label" in item for item in data))

        print("第一条数据：")
        print(data[0])
