# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN'                                                  # 模型名字
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name                          # 日志训练文件
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活，随机将一般的神经元置0
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                                          # lstm隐藏层

        self.num_layers = 3                                             # lstm层数


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
# 定义了网络神经结构
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
            # 创建一个embedding层，用来将输入数据转化为词向量
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
            # 否则，创建一个Embedding层，词汇个数为config.n_vocab，维度为config.embed，最后一个为填充向量
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                                bidirectional=True, batch_first=True, dropout=config.dropout)
            # 创建一个LSTM层，因为开启了双向模式，输出向量维度为256
            # batch_first=True指定了输入Tensor格式为(batch_size, seq_len, embed)，输出形式为(batch_size, seq_len, hidden_size * num_directions)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        # 创建一个全连接层，最后输出10个类别

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)
        # [batch_size, seq_len, embeding]=[128, 32, 300]

        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        # 句子最后时刻的 hidden state
        # LSTM层的输出，通常形状为 (batch_size, seq_len, hidden_size * num_directions)
        # out[:,-1,:]选择最后一个序列的向量，来作为全连接层的输入


        return out
