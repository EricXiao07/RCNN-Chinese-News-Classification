
"""
词表构建模块
负责将训练集中的词语转换为数字 ID。
"""

from collections import Counter

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(tokenized_texts, min_freq=1):
    """
    根据训练集分词结果构建词表。

    参数:
        tokenized_texts: 分词后的文本列表
        min_freq: 词语最低出现频率

    返回:
        vocab: 词语到数字 ID 的映射
    """
    counter = Counter()

    for tokens in tokenized_texts:
        counter.update(tokens)

    # PAD 和 UNK 固定占用前两个 ID
    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for word, count in counter.most_common():
        if count >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def text_to_ids(tokens, vocab):
    """
    将分词结果转换为数字 ID。
    不在词表中的词语使用 UNK_ID。
    """
    return [
        vocab.get(word, vocab[UNK_TOKEN])
        for word in tokens
    ]


if __name__ == "__main__":
    sample_texts = [
        ["中国", "科技", "发展"],
        ["中国", "经济", "发展"]
    ]

    vocab = build_vocab(sample_texts)

    print("词表：", vocab)
    print("转换结果：", text_to_ids(["中国", "未知词"], vocab))
