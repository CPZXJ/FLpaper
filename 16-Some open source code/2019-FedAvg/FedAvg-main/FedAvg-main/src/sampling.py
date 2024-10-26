#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users) # 用//更好
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    是non-iid,但表示极端的non-iid
    默认知道num_users的数量是100
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300 # 这个定义为自己定义，可以进行更改
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs) # 创建一个包含一系列连续整数的数组
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 生成一个两行 * 60000 列的矩阵
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()] # idx_labels[1,:]表示取数组的第二行，即包含标签信息的那一行
    idxs = idxs_labels[0, :] # 取一行的idx

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def signguard_mnist_noniid(dataset, num_users):
    """
        #暂时没有用到
        Sample non-I.I.D client data from MNIST dataset
        :param dataset: mnist数据集
        :param num_users: 有多少个用户客户端？
        :return:
        """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_class = 5
    num_imgs = int(60000 / (num_class * num_users))
    num_per_label = 6000
    dict_users = {i: np.array([]) for i in range(num_users)}  # 为每个用户客户端分配空间
    labels = dataset.targets.numpy()
    idxs = np.arange(len(labels))
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(10)]
    # number of imgs has allocated per label
    label_used = [0 for i in range(10)]

    # divide and assign 2 shards/client
    np.random.seed(2021)
    for i in range(num_users):
        rand_label = np.random.choice(label_list, num_class, replace=False)
        for y in rand_label:
            start = y * num_per_label + label_used[y]
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start + num_imgs]), axis=0)
            label_used[y] = label_used[y] + num_imgs
            if label_used[y] == num_per_label:  # 该类已经分配完了所有样本，则移除
                label_list.remove(y)
    return dict_users



def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = signguard_mnist_noniid(dataset_train, num)

