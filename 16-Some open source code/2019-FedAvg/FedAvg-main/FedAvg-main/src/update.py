#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        # 加载数据集
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                      batch_size=self.args.local_bs, shuffle=True, num_workers=4)

    def update_weights(self, model):
         # 模型训练
        model.train()
         # 定义优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9)
       #调试一下结果

        # model.zero_grad() # 在外面和里面的区别，感觉应该是在外面

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.cuda(), labels.cuda() # 得到对应的图像和labels放在GPU上进行加速计算
                model.zero_grad() # 初始化所有本地模型为零，这个是关键
                output = model(images) # 通过前向传播计算模型的输出
                loss = F.cross_entropy(output, labels) # 使用交叉熵预测损失值
                loss.backward() # 进行反向传播，计算损失相对于模型参数的梯度
                optimizer.step() # 更加梯度信息更新模型的参数
        return model.state_dict() # 返回更新后的模型参数
