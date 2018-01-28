#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/20 15:03
# @Author  : Dicey
# @Site    : Soochow
# @File    : TrainProcess.py
# @Software: PyCharm

import csv
import numpy as np
import cv2
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers import Conv2D, Cropping2D, MaxPooling2D


def readData(filePath):
    lines = []
    with open(filePath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return lines


def balanceData():
    '''
    Q1：D列Steering Angle大部分都是0，这很正常，因为大部分情况下在拐弯的地方我们才会转动方向盘。然而这对模型却是致命的，因为会造成严重的数据不平衡。
    A：随机抽样，尽量使数据平衡。在实际操作中，可以将整个Steering Angle范围划分成n个bucket(离散化)，保证每个bucket中数据样本不超过m个。
    :return: 平衡过的数据组
    '''
    evaluateList = []  # 评估列表存储转向角度不同的数量,调试用

    lines = np.array(readData('E:\AutoDriveData\driving_log.csv'))
    nbins = 2000  # 方向梯度函数
    maxExamples = 200  # 每个bucket的最大个数
    balancedArray = np.empty([0, lines.shape[1]], dtype=lines.dtype)  # lines.shape[1]列数,dtype转型
    # print(balancedArray)

    for i in range(0, nbins):
        begin = i * (1.0 / nbins)  # 区间上限
        end = begin + 1.0 / nbins  # 区间下限

        # 内部判断逻辑:其第4(D)列在区间内
        extracted = lines[(abs(lines[:, 3].astype(float)) >= begin) & (abs(lines[:, 3].astype(float)) < end)]
        if len(extracted) > 0:
            evaluateList.append(len(extracted))
        np.random.shuffle(extracted)  # 随机划分数据集
        extracted = extracted[0:maxExamples, :]  # 每组数据集取其前最大数量个
        balancedArray = np.concatenate((balancedArray, extracted), axis=0)  # 数组拼接

    # return balancedArray, evaluateList
    return balancedArray


def prepareImgs():
    ''''
    以中间摄像头照片为主训练数据，对左右两边照片的转向角度进行修正。最简单的修正方法是对左边图片的转向角度+0.2，对右边图片的转向角度-0.2。
    '''
    imgs = []
    angels = []
    offset = 0.2
    correction = [0, offset, -offset]  # 中间 左 右
    balancedArray = balanceData()
    for line in balancedArray:
        for i in range(3):
            imgPath = line[i]
            # print(imgPath)
            img = cv2.imread(imgPath)  # OpenCV处理图片
            imgs.append(img)
            # 修正角度
            angel = float(line[3])
            angels.append(angel + correction[i])

    return imgs, angels


def augmentImgs():
    ''''
    将图片进行左右反转，同时将转向角度取相反数，这样就生成了新的数据。
    再做成矩阵
    '''
    flipImgs = []
    flipAngels = []
    imgs, angels = prepareImgs()
    for img, angle in zip(imgs, angels):
        flipImgs.append(cv2.flip(img, 1))  # 图片翻转
        flipAngels.append(-1.0 * angle)  # 角度翻转

    augmentedImgs = imgs + flipImgs
    augmentedAngels = angels + flipAngels

    X_train = np.array(augmentedImgs)
    y_train = np.array(augmentedAngels)
    return X_train, y_train


def trainModel():
    X_train, y_train = augmentImgs()

    # Build the model
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))  # ??
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))  # 对2D图像进行裁剪

    model.add(Conv2D(8, (3, 3), activation='relu'))  # Covolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))  # 在训练过程中每次更新参数时随机断开一定百分比(p),防止过拟合

    model.add(Conv2D(16, (3, 3), activation='relu'))  # Covolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu'))  # Covolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())  # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。

    model.add(Dense(512, activation='relu'))  # fully connected layer
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu'))  # fully connected layer
    model.add(Dropout(0.2))

    model.add(Dense(1))  # 输出层size为1
    print("Prepare Model Comple,training...")
    # 使用linear activation对steering angle做regression。
    model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0001))  # 编译,设置损失函数与优化器

    # 回调函数 save_best_only=True意味着将只保存验证集上性能最好的模型
    best_model = ModelCheckpoint('model_best.h5', verbose=2, save_best_only=True)

    # 参数依次为:特征属性 标签 以0.2的比例划分训练集测试集 选择打乱样本 训练30次 选择回调函数
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=30, callbacks=[best_model])

    model.save('model_last.h5')


if __name__ == '__main__':
    # for line in readData('E:\AutoDriveData\driving_log.csv'):
    #     print(line)
    # a, evaList = balanceData()
    # evaList.sort()
    # for i in evaList:
    #     print(i)
    trainModel()
