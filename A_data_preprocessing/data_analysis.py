
"""
TNEWS 数据集统计与分析模块
负责统计数据数量、类别分布和文本长度等信息。
"""


def analyze_data(data):
    """
    分析数据集的基本情况。

    参数:
        data: 数据列表，每条数据包含 sentence 和 label。
    """
    print(f"数据总量：{len(data)}")

    label_counts = {}
    for item in data:
        label = item.get("label", "未知")
        label_counts[label] = label_counts.get(label, 0) + 1

    print("各类别数据数量：")
    for label, count in sorted(label_counts.items()):
        print(f"类别 {label}: {count} 条")
