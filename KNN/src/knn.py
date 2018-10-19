#!/usr/bin/env python
# encoding: utf-8
'''
@author: TszFung_Chan
@contact: ex_lunatic@hotmail.com
@software: TszFung_Chan
@file: knn
@time: 2018/10/12 21:34
@desc: KNN algorithm
'''

import numpy as np
import matplotlib.pyplot as plt

# create some data as reference data


def create_random_data():
    # random 100 pairs of (x, y), make the divide them by x / 25
    datax = np.round(np.random.rand(100, 1) * 100)
    datay = np.round(np.random.rand(100, 1) * 100)
    datax.sort(axis=0)
    return datax, datay

# draw the reference data


def draw_fix(x, y, color):
    plt.scatter(x, y, s=20, c=color, marker="x", alpha=1)


def painting(datax, datay):
    # draw the fix point
    draw_fix(datax[0: 25], datay[0: 25], "red")
    draw_fix(datax[25: 50], datay[25: 50], "green")
    draw_fix(datax[50: 75], datay[50: 75], "blue")
    draw_fix(datax[75: 100], datay[75: 100], "brown")
    plt.show()

    # use a map array to record the reference point
    # to create a 100*110 matrix represent the map
    map = [[False for col in range(110)] for row in range(110)]
    for i in range(100):
        map[int(datax[i])][int(datay[i])] = True

    # use KNN to evaluate the unkonwn zone
    for i in range(100):
        for j in range(100):
            if not map[i][j]:
                predict(i, j)

# KNN, we use 5 reference point here, and the nearer one has a greater weight by using 100 / dis


def predict(x, y):
    # 4 classes and their score
    color = ["red", "green", "blue", "brown"]
    score = np.array(np.zeros(4))
    tar = np.array([x, y])
    dis = []

    # evaluate the distance for the 100 reference point
    for i in range(100):
        vector = np.array([int(datax[i]), int(datay[i])])
        dist = np.linalg.norm(vector - tar)
        dis.append([dist, int(i / 25)])
    dis = sorted(dis, key=lambda entry: entry[0])

    # the score of the nearest 5 reference points
    for i in range(5):
        score[int(dis[i][1])] += 100 / dis[i][0]

    # decide the class
    which = 3
    highest = score[3]
    for i in range(3):
        if score[i] > highest:
            highest = score[i]
            which = i

    # draw it on the plot
    plt.scatter(np.array([x]), np.array([y]), s=10, c=color[which], alpha=1)



datax, datay = create_random_data()
painting(datax, datay)
plt.show()

