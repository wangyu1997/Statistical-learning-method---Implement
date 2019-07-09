#!encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import random


class Perceptron:
    train_data = []
    learning_rate = 0.01
    train_num = 1
    gram = np.array([])
    a = np.array([])
    N = 0
    b = 0
    w = np.array([0.0, 0.0])

    def __init__(self, train_data, train_num, learning_rate):
        self.train_data = train_data
        self.train_num = train_num
        self.learning_rate = learning_rate
        self.N = len(self.train_data)
        self.gram = [[0.0 for c in range(self.N)] for r in range(self.N)]
        self.a = np.array([0.0 for i in range(self.N)])
        self.b = 0.0

    def disc_sign(self, i):
        y_i = self.train_data[i][-1]
        res = 0.0
        for j in range(self.N):
            y_j = self.train_data[j][-1]
            a_j = self.a[j]
            x_ji = self.gram[j][i]
            res += a_j * y_j * x_ji
        res += self.b
        res *= y_i
        return res <= 0.0

    def calc_w(self):
        for i in range(self.N):
            x_i = self.train_data[i][:2]
            y_i = self.train_data[i][-1]
            self.w += self.a[i] * y_i * x_i


    def get_false_list(self):
        list = []
        for i in range(self.N):
            if self.disc_sign(i):
                list.append(i)
        return list

    def train(self):
        self.init_gram()
        plt.close()
        plt.figure()
        self.draw_data()
        for i in range(self.train_num):
            false_list = self.get_false_list()
            if len(false_list)==0:
                break
            index = false_list[random.randint(0, len(false_list) - 1)]
            self.a[index] += self.learning_rate
            self.b += self.learning_rate * self.train_data[index][-1]
        self.calc_w()
        self.draw_line(self.w, self.b)
        plt.show()

    def init_gram(self):
        train_len = len(self.train_data)
        for i in range(train_len):
            for j in range(train_len):
                x_i = np.array(self.train_data[i])[0:2]
                x_j = np.array(self.train_data[j])[0:2]
                self.gram[i][j] = np.dot(x_i, x_j)

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
        delta = 0.0000000000001 #纠正参数
        if w[1] is 0.0:
            w[1] += delta
        x2 = (-b - w[0] * x1) / w[1]
        plt.plot(x1, x2, c='r')


def main():
    train_data1 = [[1, 3, 1], [2, 2, 1], [3, 8, 1], [2, 6, 1], [3, 3, 1], [3, 4, 1], [5, 5, 1]]  # 正样本
    train_data2 = [[2, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1], [3, 1, -1], [4, 1, -1], [5, 2, -1]]  # 负样本
    train_data = np.array(train_data1 + train_data2)  # 合并正负样本数据
    # train_data = np.array([[3, 3, 1], [4, 3, 1], [1, 1, -1]])
    p = Perceptron(train_data, 100, 0.01)
    p.train()


if __name__ == '__main__':
    main()
