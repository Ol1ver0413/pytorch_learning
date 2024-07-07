# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
# 初始化神经网络模型，传入参数为model（模型），method（权重初始化参数），exclude（排除初始化参数），seed（固定随机种子）
    for name, w in model.named_parameters():
        if exclude not in name:
        # 代表不初始化embedding参数
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                    # 'xavier' 使用 nn.init.xavier_normal_ 方法，适合深度网络的权重初始化

                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
                # 如果参数名称中包含 'bias'，则将其初始化为常数0
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter,writer):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 定义一个Adam优化器，传入的为需要更新的参数，lr代表学习率

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    # 设置最佳损失为无穷大，确保之后的损失都比它小

    last_improve = 0
    # 记录上次验证集loss下降的batch数

    flag = False
    # 记录是否很久没有效果提升
    #writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
        # enumerate(train_iter) 用于获取每个批次的索引 i 和内容。内容通常包括特征数据 trains 和对应的标签 labels
            print (trains[0].shape)
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            # 计算梯度
            optimizer.step()
            # 更新参数

            if total_batch % 100 == 0:
            # 监控和记录
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                # 将labels数据移动到cpu上执行

                predic = torch.max(outputs.data, 1)[1].cpu()
                # 返回预测最大值，因为是[batch_size,num_classes]，返回第一个维度上的

                train_acc = metrics.accuracy_score(true, predic)
                # 计算准确率
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    # 保存模型最好的状态时候的权重参数，并保存到path文件中

                    improve = '*'
                    # 记录，模型是否有所提升
                    last_improve = total_batch
                    # last_improve 设置为 total_batch 的值意味着我们正在记录最后一次模型性能提升时的批次编号
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)
    # 开启测试


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    # 验证集或测试集上评估模型的性能

    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
# 验证集或测试集上评估模型的性能

    model.eval()
    loss_total = 0
    # 用于累积所有批次的总损失
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # 存储所有的预测结果和真实结果

    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    # 计算测试集的准确率

    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        # 打印一份报告，包括真实标签，预测标签，以及类别的名称列表
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        # 计算混淆矩阵
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)