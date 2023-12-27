import jieba
from tqdm import tqdm


# 加载停用词
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return stopwords


# 对数据集进行分词，加载停用词
def word_split(texts):
    words = []
    stopwords = load_stopwords('./cn_stopwords.txt')
    for text in tqdm(texts):
        word = jieba.cut(text)
        words.append(' '.join(w for w in word if w not in stopwords and w != ' '))
    return words
