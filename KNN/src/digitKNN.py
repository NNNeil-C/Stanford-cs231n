#!/usr/bin/env python
# encoding: utf-8
'''
@author: TszFung_Chan
@contact: ex_lunatic@hotmail.com
@software: TszFung_Chan
@file: digitKNN
@time: 2018/10/14 16:43
@desc: use KNN to recognize handwriting digits
'''

import os
import numpy as np


def loadDataSet(dir):
    print ("loading Dataset...")
    # first, get all the filename in the directory
    file_list= os.listdir(dir)
    # number of file in the directory
    file_number = len(file_list)
    # 1024 pixel * file_number
    data = np.zeros((file_number, 1024))
    label = np.zeros(file_number)

    for idx, filename in enumerate(file_list):
        data[idx], label[idx] = txt2vector(dir, filename)

    return data, label


def txt2vector(dir, filename):
    # get the label of this image
    label = filename.split('_')[0]
    # the target vector
    data = np.zeros(1024)

    # open file
    with open(dir + "\\" + filename, 'r') as f:
        # read by line
        for row in range(32):
            str = f.readline()
            for col in range(32):
                data[row * 32 + col] = str[col]

    return data, label




def predict(testSet, testLabel):
    success = 0
    i = 1
    for candidate, ans in zip(testSet, testLabel):
        label = KNNClassifier(candidate, reference=100)

        if label == ans:
            success += 1
        else:
            print(candidate, label, ans, i)
        i += 1

    rate = success / len(testLabel)

    print("Using KNN-{number}, the accuracy is {rate}% for {space}".format(number=100, rate=rate * 100, space=len(testLabel)))


def KNNClassifier(candidate, reference):
    dis = []
    score = list(np.zeros(10))
    for i in range(len(trainningSet)):
        dist = np.linalg.norm(trainningSet[i] - candidate)
        dis.append([dist, trainLabel[i]])
    dis = sorted(dis, key=lambda entry: entry[0])

    # the score of the nearest reference digits
    for i in range(reference):
        score[int(dis[i][1])] += 100 / dis[i][0]

    # decide the class
    which = 9
    highest = score[9]
    for i in range(9):
        if score[i] > highest:
            highest = score[i]
            which = i

    return which



trainningSet, trainLabel = loadDataSet(r'..\739988digits\trainingDigits')
testSet, testLabel = loadDataSet(r'..\739988digits\testDigits')

predict(testSet, testLabel)

