
from pathlib import Path
from load_data import load_jsonl
from tokenizer import tokenize

DATA_DIR = Path("data/raw/TNEWS")


def analyze(name):
    data = load_jsonl(DATA_DIR / f"{name}.json")

    lengths = [
        len(tokenize(item["sentence"]))
        for item in data
    ]

    sorted_lengths = sorted(lengths)

    def percentile(p):
        index = int((len(sorted_lengths) - 1) * p)
        return sorted_lengths[index]

    over_50 = sum(length > 50 for length in lengths)
    over_100 = sum(length > 100 for length in lengths)

    print(f"\n===== {name} 文本长度统计 =====")
    print("样本数量：", len(lengths))
    print("最短长度：", min(lengths))
    print("最长长度：", max(lengths))
    print("平均长度：", round(sum(lengths) / len(lengths), 2))
    print("中位数：", percentile(0.5))
    print("90%分位数：", percentile(0.9))
    print("95%分位数：", percentile(0.95))
    print("99%分位数：", percentile(0.99))
    print("超过50词的数量：", over_50)
    print("超过50词的比例：", round(over_50 / len(lengths) * 100, 2), "%")
    print("超过100词的数量：", over_100)


if __name__ == "__main__":
    for name in ["train", "dev", "test"]:
        analyze(name)
