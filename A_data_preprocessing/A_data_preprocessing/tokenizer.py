
"""
中文分词模块
负责将新闻标题切分为词语序列。
"""

import jieba


def tokenize(text):
    """
    对中文新闻标题进行分词。

    参数:
        text: 新闻标题字符串

    返回:
        分词后的词语列表
    """
    if not isinstance(text, str):
        return []

    return list(jieba.cut(text))


if __name__ == "__main__":
    test_text = "中国科学家取得重大科技突破"
    print("原始文本：", test_text)
    print("分词结果：", tokenize(test_text))
