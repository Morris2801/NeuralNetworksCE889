from neuron import Neuron, Layer, Network
import random as rd
import matplotlib.pyplot as plt

##setup---------------------------
alpha = 0.15 # momentum
lambHidden = 0.8 # labmda
lambOutput = 0.75
biasHidden = 0
biasOutput = 0
eta = 0.1 #learning rate

##newtork
inputLayer = Layer(2)
hiddenLayer = Layer(4, activation_fn=Neuron.activation_f_sigmoid, lamb=lambHidden,bias= biasHidden)
outputLayer = Layer(1, bias=biasOutput)

net = Network(inputLayer, hiddenLayer, outputLayer, eta, alpha)
for i in range(len(net.layers)-1):
    net.fullyConnect(net.layers[i], net.layers[i+1])

datasetXOR = [
    ([0,0],[0]),
    ([0,1],[1]),
    ([1,0],[1]),
    ([1,1],[0])
]
datasetAverage = [
    ([1, 6],  [3.5]),
    ([2, 4],  [3]),
    ([3, 7],  [5]),
    ([4, 8],  [6]),
    ([5, 9],  [7]),
    ([6, 12], [9]),
    ([7, 10], [8.5]),
    ([8, 16], [12]),
    ([9, 11], [10]),
    ([10, 13],[11.5]),
    ([12, 15],[13.5]),
    ([15, 20],[17.5]),
    ([20, 30],[25]),
]
datasetSumSubst = [
    ([1, 6],  [7, -5]),
    ([2, 4],  [6, -2]),
    ([3, 7],  [10, -4]),
    ([4, 8],  [12, -4]),
    ([5, 9],  [14, -4]),
    ([6, 12], [18, -6]),
    ([7, 10], [17, -3]),
    ([8, 16], [24, -8]),
    ([9, 11], [20, -2]),
    ([10, 13],[23, -3]),
    ([12, 15],[27, -3]),
    ([15, 20],[35, -5]),
    ([20, 30],[50, -10]),
]
datasetIdentity = [
    ([0, 0],  [0]),
    ([1, 1],  [1]),
    ([3, 3],  [3]),
    ([4, 4],  [4]),
    ([5, 5],  [5]),
    ([10, 10], [10]),
    ([7, 7], [7]),
    ([8, 8], [8]),
    ([11, 11], [11]),
    ([-8, -8], [-8]),
    ([12, 12], [12]),
    ([25, 25], [25]),
    ([30, 30], [30]),
]

net.train(datasetXOR,10000)

epochs = []
sses = []
for epochResults in net.trainingSSELog:
    epochs.append(epochResults[0])
    sses.append(epochResults[1])
plt.plot(epochs, sses)
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("Training SSE")
plt.show()

target = 0
prediction = net.predict([1,1])
print(f"Target {target} vs Prediction {prediction[0]}")