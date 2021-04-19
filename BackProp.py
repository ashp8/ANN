import numpy as np
import math
import random as rand

epoch = 10000
mean = 0
exp_mean = 0.0
lr = 0.01 ### learning rate
xi = 7 ### input units
xh = 5 ### hidden units
xy = 1

### Network Weights
w1 = np.random.random([xi, xh - 1])
w2 = np.random.random([xh])

### Target Weights
target_weight_1 = np.random.random([xi, xh - 1])
target_weight_2 = np.random.random([xh])

### Training Session
for i in range(0, epoch):
    x = [1]
    s1 = 0
    for j in range(0, xi - 1):
        val = 1 if (rand.random() > 0.5) else -1
        x.append(val)
        s1 += val

    x = np.array(x, dtype = "int64")

    ### Forward Pass
    x_h = [1]
    x_ht = [1]
    h = np.dot(x, w1)
    ht = np.dot(x, target_weight_1)
    for j in range(0, xh - 1):
        h[j] = math.tanh(h[j])
        x_h.append(h[j])
        ht[j] = math.tanh(ht[j])
        x_ht.append(ht[j])

    x_h = np.array(x_h)
    x_ht = np.array(x_ht)
    y = np.dot(x_h, w2)
    y = math.tanh(y)
    t = np.dot(x_ht, target_weight_2)
    t = math.tanh(t)

    ## loss evaluation (SE)
    loss = 1/2 * (t - y) ** 2
    mean = mean + loss
    exp_mean = (exp_mean * 0.99) + (loss * 0.01)

    ## Backprop
    m1 = -1 * (t - y) * (1 - y ** 2)
    dw2 = []

    for j in range(0, xh):
        dw2.append(m1 * x_h[j])

    dw2 = np.array(dw2)
    dw1 = []
    for j in range(1, xh):
        dw = []
        for k in range(0, xi):
            dmw = m1 * w2[j] * (1 - x_h[j] ** 2) * x[k]
            dw.append(dmw)
        dw1.append(dw)

    ## Weight Update
    dw1 = np.array(dw1)
    dw1 = np.transpose(dw1)
    w1 -= dw1 * lr
    w2 -= dw2 * lr

    if(i % 100 == 0):
        print("=== X: %s ===" %(x,))
        print("Y: %s " %(y,))
        print("T: %s " %(t,))
        print("LOSS: %s " %(loss,))
        print("Running mean: %s " %(exp_mean,))
        print("MSL: %s " %(mean/(i + 1),))