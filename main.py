import math
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
#y = [[0], [0], [0], [1]] #and gate
y = [[1], [1], [1], [0]] #or gate
lr = 0.20
b = -1
bw = 0.3
w = [0.4, 0.5]

def sigmoid(x):
    return 1.0 / (1.0 - math.exp(-x));

for _ in range(500):
    for i in range(0, len(x)):
        z = x[i][1] * w[0] + x[i][1] * w[1] + b * bw
        err = y[i][0] - z
        nw1 = lr * err * x[i][0]
        nw2 = lr * err * x[i][1]
        w[0] += nw1
        w[1] += nw2
        bw += lr * err * b
        print("===================")
        print(f"[{_}]Error:{err}")
        print(f"weight: [{w[0]}, {w[1]}]")
        print(f"z:{z}")
        print("===================")

def predict(arr):
    z = arr[0] * w[0] + arr[1]*w[1] + b * bw
    print(z)
    if(z > 0.1):
        print("output: 1")
    else:
        print("output: 0")
predict([1, 0]);
# andgate()
