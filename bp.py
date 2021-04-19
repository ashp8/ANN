import math
import random

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def makeMatrix(i, j, fill=0.0):
    m = []
    for i in range(i):
        m.append([fill] * j)
    return m


def sigmoid(x):
    return math.tanh(x)
    # return (1.0)/(1.0-math.exp(-x))


def dsigmoid(x):
    return 1.0 - x ** 2


class NN:
    def __init__(self, il, hl, ol):
        self.il = il + 1
        self.hl = hl
        self.ol = ol

        self.ai = [1.0] * self.il
        self.ah = [1.0] * self.hl
        self.ao = [1.0] * self.ol

        self.wi = makeMatrix(self.il, self.hl)
        print(self.wi)
        self.wo = makeMatrix(self.hl, self.ol)
        
        for i in range(self.il):
            for j in range(self.hl):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.hl):
            for k in range(self.ol):
                self.wo[j][k] = rand(-0.2, 2.0)

        # last change in weights for speed up
        self.ci = makeMatrix(self.il, self.hl)
        self.co = makeMatrix(self.hl, self.ol)

    def update(self, inputs):
        if len(inputs) != self.il - 1:
            raise ValueError("Wrong number of inputs: ")

        for i in range(self.il - 1):
            self.ai[i] = inputs[i]

        for j in range(self.hl):
            sum = 0.0
            for i in range(self.il):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)
        for k in range(self.ol):
            sum = 0.0
            for j in range(self.hl):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)
        return self.ao[:]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.ol:
            raise ValueError("Wrong number of target values")
        output_deltas = [0.0] * self.ol
        for k in range(self.ol):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        hidden_deltas = [0.0] * self.hl
        for j in range(self.hl):
            error = 0.0
            for k in range(self.ol):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        for j in range(self.hl):
            for k in range(self.ol):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change

        for i in range(self.il):
            for j in range(self.hl):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        for i in patterns:
            print(i[0], '->', self.update(i[0]))

    def weights(self):
        print('Input Weights: ')
        for i in range(self.il):
            print(self.wi[i])
        print()
        print('Output weights: ')
        for j in range(self.hl):
            print(self.wo[j])

    def train(self, patterns, it=1000, N=0.5, M=0.1):
        for i in range(it):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)


def inp():
    data = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]],
    ]
    n = NN(2, 2, 1)
    # n = NN(3, 1)
    n.train(data)
    n.test(data)


if __name__ == '__main__':
    inp()
