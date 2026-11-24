
from collections import Counter
from load_data import load_jsonl

def analyze_data(data):
    """统计新闻数据集的类别分布"""

    label_counts = Counter(
        item["label"] for item in data
        if "label" in item
    )

    # 获取类别编号与类别名称的对应关系
    label_names = {}
    for item in data:
        if "label" in item:
            label_names[item["label"]] = item.get(
                "label_desc", "未知类别"
            )

    total = sum(label_counts.values())

    print("\n===== 新闻类别分布统计 =====")
    print(f"带标签的数据总量：{total}")
    print(f"新闻类别数量：{len(label_counts)}")
    print("-" * 45)

    for label, count in sorted(label_counts.items()):
        name = label_names.get(label, "未知类别")
        percentage = count / total * 100

        print(
            f"类别编号：{label} | "
            f"类别名称：{name} | "
            f"数量：{count} | "
            f"占比：{percentage:.2f}%"
        )

if __name__ == "__main__":

    train_path = "data/raw/TNEWS/train.json"

    train_data = load_jsonl(train_path)

    analyze_data(train_data)
