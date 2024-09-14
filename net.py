import numpy as np
import random
import copy


class network:
    inputSize = 28 * 28
    numH = 2
    hSize = 16
    outputSize = 10

    weights = list()
    biases = list()

    def __init__(self, inputSize, numHidden, hiddenSize, outputSize):
        self.inputSize = inputSize
        self.numH = numHidden
        self.hSize = hiddenSize
        self.outputSize = outputSize

        self.weights = self.getInitWeights()
        self.biases = self.getInitBiases()

    def getInitWeights(self):
        weightList = list()

        for layer in range(0, self.numH + 1):

            if layer == 0:
                rowSize = self.inputSize
            else:
                rowSize = self.hSize

            if layer == self.numH:
                colSize = self.outputSize
            else:
                colSize = self.hSize

            weightMat = np.zeros((colSize, rowSize))

            for row in range(0, colSize):
                for col in range(0, rowSize):
                    weightMat[row][col] = random.uniform(-0.5, 0.5)

            weightList.append(weightMat)

        return weightList

    def getInitBiases(self):
        biasList = list()

        for layer in range(0, self.numH + 1):

            if layer == self.numH:
                size = self.outputSize
            else:
                size = self.hSize

            biasVec = np.zeros(size)

            for row in range(0, size):
                biasVec[row] = random.uniform(-0.5, 0.5)

            biasList.append(biasVec)

        return biasList

    def forwardProp(self, input):
        output = input
        activations = list()
        z = list()
        activations.append(np.copy(output))

        for layer in range(0, self.numH):
            output = np.matmul(self.weights[layer], output)
            output = np.add(output, self.biases[layer])
            z.append(np.copy(output))
            output = np.vectorize(network.Sigmoid)(output)
            activations.append(np.copy(output))

        output = np.matmul(self.weights[self.numH], output)
        output = np.add(output, self.biases[self.numH])
        z.append(np.copy(output))
        output = network.Softmax(output)
        activations.append(np.copy(output))
        return activations, z

    def cost(output, answer):
        sum = 0
        for i in range(0, len(output)):
            sum += np.power((output[i] - (1 if i == answer else 0)), 2)
        return sum

    def getNabla(self, input, answer):
        activations, z = self.forwardProp(input)
        weightsList = copy.deepcopy(self.weights)
        biasesList = copy.deepcopy(self.biases)

        dcdaPrev = list()

        # output layer
        for row in range(0, self.outputSize):
            dcda = 2 * (activations[-1][row] - (1 if row == answer else 0))
            dadz = network.dSoftmax(activations[-1][row])
            for col in range(0, self.hSize):
                dzdw = activations[-2][col]
                weightsList[-1][row][col] = dzdw * dadz * dcda
            biasesList[-1][row] = dadz * dcda

        for k in range(0, self.hSize):
            sum = 0

            for j in range(0, self.outputSize):
                dcda = 2 * (activations[-1][j] - (1 if j == answer else 0))
                dadz = network.dSoftmax(activations[-1][j])
                dcdak = self.weights[-1][j][k]
                sum = sum + dcdak * dadz * dcda

            dcdaPrev.append(sum)
            sum = 0

        # hidden layers
        for layer in range(-2, -(self.numH + 1), -1):
            for row in range(0, self.hSize):
                dcda = dcdaPrev[row]
                dadz = network.dSigmoid(z[layer][row])
                for col in range(0, self.hSize):
                    dzdw = activations[layer - 1][col]
                    weightsList[layer][row][col] = dzdw * dadz * dcda
                biasesList[layer][row] = dadz * dcda

            dcdaPrevNew = list()

            for k in range(0, self.hSize):
                sum = 0

                for j in range(0, self.hSize):
                    dcda = dcdaPrev[j]
                    dadz = network.dSigmoid(activations[layer][j])
                    dcdak = self.weights[layer][j][k]
                    sum = sum + dcdak * dadz * dcda

                dcdaPrevNew.append(sum)
                sum = 0
            dcdaPrev = dcdaPrevNew

        # first hidden
        for row in range(0, self.hSize):
            dcda = dcdaPrev[row]
            dadz = network.dSigmoid(z[0][row])
            for col in range(0, self.inputSize):
                dzdw = input[col]
                weightsList[0][row][col] = dzdw * dadz * dcda
            biasesList[0][row] = dadz * dcda

        return self.getWeightBiasVec(weightsList, biasesList), network.cost(
            activations[-1], answer
        )

    def learn(self, nabla, learnRate):
        wbVec = self.getNetWeightBiasVec()
        wbVec = np.add(np.multiply(nabla, -learnRate), wbVec)
        self.initWeightsBiases(wbVec)
        return

    def ReLU(i):
        return np.maximum(0, i)

    def Sigmoid(n):
        return 1 / (1 + np.exp(-n))

    def dSigmoid(n):
        return np.exp(-n) / np.power((1 + np.exp(-n)), 2)

    def Softmax(vector):
        max = np.max(vector)
        adjustedVec = np.subtract(vector, max)
        sum = np.sum(np.exp(adjustedVec))
        return np.exp(adjustedVec) / sum

    def dSoftmax(n):
        return n * (1 - n)

    def getyVec(n):
        arr = np.zeros(10)
        arr[n - 1] = 1
        return arr

    # **************** Util *******************
    def getNetWeightBiasVec(self):
        vec = list()
        # weights
        for wMat in self.weights:
            for wRow in wMat:
                for w in wRow:
                    vec.append(w)
        for bVec in self.biases:
            for b in bVec:
                vec.append(b)
        return np.array(vec)

    def getWeightBiasVec(self, weights, biases):
        vec = list()
        # weights
        for wMat in weights:
            for wRow in wMat:
                for w in wRow:
                    vec.append(w)
        for bVec in biases:
            for b in bVec:
                vec.append(b)
        return np.array(vec)

    def initWeightsBiases(self, wbVec):
        ind = 0

        # read input weights
        for row in range(0, self.hSize):
            for col in range(0, self.inputSize):
                self.weights[0][row][col] = wbVec[ind]
                ind += 1

        # read hidden layer weights
        for hLayer in range(1, self.numH):
            for row in range(0, self.hSize):
                for col in range(0, self.hSize):
                    self.weights[hLayer][row][col] = wbVec[ind]
                    ind += 1

        # read pre output layer weights
        for row in range(0, self.outputSize):
            for col in range(0, self.hSize):
                self.weights[self.numH][row][col] = wbVec[ind]
                ind += 1

        # read hidden layer biases
        for layer in range(0, self.numH):
            for row in range(0, self.hSize):
                self.biases[layer][row] = wbVec[ind]
                ind += 1

        # read output layer biases
        for row in range(0, self.outputSize):
            self.biases[self.numH][row] = wbVec[ind]
            ind += 1


net = network(1, 2, 1, 1)
print(net.getNetWeightBiasVec())
net.getNabla(np.array([0]), 0)
print(net.getNetWeightBiasVec())

# for i in range(0, 5):
#     net.learn(net.getNabla(np.array([0.5]), 1), 0)
#     print(net.weights)


# print(net.inputSize, net.numH, net.hSize, net.outputSize)

# print(net.weights, "\n")
# print(net.biases, "\n")

# np.save("mnist_16x16_model", net.getWeightBiasVec())

# wbVec = np.load('mnist_16x16_model.npy')
# print(wbVec,"\n")
# net.initWeightsBiases(wbVec)

# print(net.weights, "\n")
# print(net.biases, "\n")
