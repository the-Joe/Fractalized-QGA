import numpy as np


class NeuralNetworkXOR:

    def __init__(self, numHidden, numPasses, learningRate):
        self.inpDim = 2
        self.outDim = 1
        self.numHidden = numHidden
        self.numPasses = numPasses
        self.learningRate = learningRate
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float64)
        self.y = np.array([[0], [1], [1], [0]], np.float64)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidPrime(self, x):
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

    def initializeWeights(self):
        self.w1 = np.random.rand(self.inpDim, self.numHidden)
        self.b1 = np.random.uniform(-0.5, 0.5, size=self.numHidden)
        self.w2 = np.random.rand(self.numHidden, self.outDim)
        self.b2 = np.random.uniform(-0.5, 0.5, size=self.outDim)

    def forwardPropagate(self):
        self.z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.yHat = self.sigmoid(self.z2)
        self.cost = 0.5 * np.sum(np.square(self.y - self.yHat))

    def backPropagate(self):
        self.delta2 = np.multiply(-(self.y - self.yHat), self.sigmoidPrime(self.z2))
        self.dcdw2 = np.dot(self.a1.T, self.delta2)
        self.delta1 = np.dot(self.delta2, self.w2.T) * self.sigmoidPrime(self.z1)
        self.dcdw1 = np.dot(self.x.T, self.delta1)

    def updateWeights(self):
        self.w1 -= self.dcdw1 * self.learningRate
        self.b1 -= np.mean(self.dcdw1) * self.learningRate
        self.w2 -= self.dcdw2 * self.learningRate
        self.b2 -= np.mean(self.dcdw2) * self.learningRate

    def train(self):
        self.initializeWeights()
        for i in range(self.numPasses):
            self.forwardPropagate()
            self.backPropagate()
            self.updateWeights()
            if i % 5000 == 0:
                print("Iteration {i} \nOutput:\n {y} \
                                       \nPredicted Output:\n {yHat} \nCost:\n{cost} \n"
                      .format(i=i, y=self.y, \
                              yHat=self.yHat, \
                              cost=self.cost))


def main():
    nn = NeuralNetworkXOR(numHidden=3, \
                          numPasses=25000, \
                          learningRate=1)
    nn.train()


if __name__ == '__main__':
    main()
