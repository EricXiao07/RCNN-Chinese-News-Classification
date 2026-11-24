
import jieba


def tokenize(text):
    """
    对中文新闻文本进行分词
    """
    if not isinstance(text, str):
        return []

    words = jieba.cut(text)
    return list(words)


if __name__ == "__main__":

    sentence = "苹果发布新款手机"

    result = tokenize(sentence)

    print("原始文本：", sentence)
    print("分词结果：", result)
    print("分词数量：", len(result))
