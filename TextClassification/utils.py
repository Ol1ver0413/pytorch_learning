# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000
# 词表长度限制

UNK, PAD = '<UNK>', '<PAD>'
# UNK代表未知字，PAD代表填充字


def build_vocab(file_path, tokenizer, max_size, min_freq):


    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
# 建立词汇表
    if ues_word:
        tokenizer = lambda x: x.split(' ')
        # 按词分类，以空格隔开，word-level

    else:
        tokenizer = lambda x: [y for y in x]
        # 按字分类

    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        # 创建词汇表

        pkl.dump(vocab, open(config.vocab_path, 'wb'))
        # 将词汇表保存到指定路径

    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
    # pad_size指得是文本最大长度

        contents = []
        #用于存储处理好后的文本和标签
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
            # 逐行遍历文件对象f的每一行
                lin = line.strip()
                # 去除每一行两端的空白字符
                if not lin:
                    continue
                # 如果空，就跳过

                content, label = lin.split('\t')
                # 语句和标签按照‘\t’分割

                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                # 多截少补
                    if len(token) < pad_size:
                        token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                        # list.extend(iterable)将可迭代对象追加到列表末尾
                        # dict.get(key)获取key值所对应的value

                    else:
                        token = token[:pad_size]
                        # 进行截断操作
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                    # word存在，返回word的结果；不存在返回UNK的结果

                contents.append((words_line, int(label), seq_len))
                # contents里包含了每个字所对应的ID组成的向量 + 标签 + 这句话的长度
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        # 每个训练批次的大小

        self.batches = batches
        # 包含所有数据的列表

        self.n_batches = len(batches) // batch_size
        # 批次数量

        self.residue = False
        # 记录batch数量是否为整数，是否可以被整除

        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
    # 将张量移动到指定设备

        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # 将特征向量转化为长整型张量，并放入gpu中运算

        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
        # 判断是否有多余批次以及当前索引是否为最后一个批次
        # 如果 self.index 等于 self.n_batches，说明已经处理完所有完整的批次，现在是时候处理残余批次了

            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            # self.index * self.batch_size表示为当前批次的初始索引

            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
# 构建一个迭代器
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
