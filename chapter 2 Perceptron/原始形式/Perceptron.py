#!encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import random


class Perceptron:
    train_data = []
    learning_rate = 0.01
    train_num = 1
    w = np.array([0.0,0.0]).T
    b = 0.0

    def __init__(self, train_data, train_num, learning_rate):
        self.train_data = train_data
        self.train_num = train_num
        self.learning_rate = learning_rate

    def train(self):
        plt.close()
        plt.figure()
        self.draw_data()
        for i in range(self.train_num):
            x1, x2, y = random.choice(self.train_data)
            x = np.array([x1, x2])
            if y * (np.dot(self.w, x) + self.b) <= 0:
                self.w += self.learning_rate * y * x
                self.b += self.learning_rate * y
        self.draw_line(self.w, self.b)
        plt.show()

    def draw_data(self):
        for data in self.train_data:
            y = data[2]
            if y > 0:
                plt.scatter(data[0], data[1], c='k', marker='o', s=50)
            else:
                plt.scatter(data[0], data[1], c='k', marker='x', s=50)

    @staticmethod
    def draw_line(w, b):
        x1 = np.linspace(0, 8, 100)
        x2 = (-b - w[0] * x1) / w[1]
        plt.plot(x1, x2, c='r')


def main():
    train_data1 = [[1, 3, 1], [2, 2, 1], [3, 8, 1], [2, 6, 1], [3, 3, 1], [3, 4, 1], [5, 5, 1]]  # 正样本
    train_data2 = [[2, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1], [3, 1, -1], [4, 1, -1], [5, 2, -1]]  # 负样本
    train_data = np.array(train_data1 + train_data2)  # 合并正负样本数据
    p = Perceptron(train_data, 100, 0.01)
    p.train()


if __name__ == '__main__':
    main()
