
"""
TNEWS 数据集读取模块

负责：
1. 读取 JSONL 格式的数据
2. 检查数据字段
3. 返回新闻记录列表
"""

import json
from pathlib import Path


def load_jsonl(file_path):
    """
    读取 JSONL 文件。

    每一行是一条独立的 JSON 新闻记录。

    参数:
        file_path: JSON 文件路径

    返回:
        data: 新闻记录列表
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{file_path}")

    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"第 {line_number} 行 JSON 格式错误：{e}"
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


def check_data(data):
    """
    检查数据基本情况。
    """
    print("=" * 40)
    print("TNEWS 数据检查")
    print("=" * 40)

    print(f"数据总量：{len(data)}")

    if not data:
        print("数据集为空！")
        return

    print("\n第一条数据：")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))

    print("\n字段名称：")
    print(list(data[0].keys()))

    labeled_count = sum(
        1 for item in data if item.get("label") is not None
    )

    print(f"\n有标签的数据：{labeled_count}")
    print(f"无标签的数据：{len(data) - labeled_count}")


if __name__ == "__main__":
    # 后续根据本地项目目录设置实际路径
    print("TNEWS 数据读取模块已准备完成。")
