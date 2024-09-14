from keras.api.datasets import mnist
import numpy as np
import pickle
from matplotlib import pyplot
from net import network

# loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

sampleSize = 100
learnRate = 5

trainingLength = len(train_X)

net = network(28 * 28, 3, 16, 10)
wbVec = np.load("mnist_16x16x16_model.npy")
net.initWeightsBiases(wbVec)

nablaLength = 0
for layer in range(0, net.numH + 1):
    weightShape = np.shape(net.weights[layer])
    nablaLength += weightShape[0] * weightShape[1] + len(net.biases[layer])

leftInd = 0
rightInd = 0

while leftInd < trainingLength:
    rightInd = leftInd + sampleSize
    if rightInd > trainingLength:
        break

    nabla = np.zeros(nablaLength)

    costSum = 0

    for i in range(leftInd, rightInd):
        imageVec = np.array(train_X[i]).flatten()
        imageVec = np.divide(imageVec, 255)
        nab, cost = net.getNabla(imageVec, train_y[i])
        nabla = np.add(nabla, nab)
        costSum += cost

    nabla = np.divide(nabla, (rightInd - leftInd))
    costSum = costSum / (rightInd - leftInd)

    net.learn(nabla, learnRate)
    print(rightInd, ": ", costSum)

    costSum = 0

    leftInd = rightInd

np.save("mnist_16x16x16_model", net.getNetWeightBiasVec())
