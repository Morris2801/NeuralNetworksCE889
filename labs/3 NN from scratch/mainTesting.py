from neuron import Neuron, Layer, Network
import random as rd

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
outputLayer = Layer(2, bias=biasOutput)

net = Network(inputLayer, hiddenLayer, outputLayer, eta, alpha)
for i in range(len(net.layers)-1):
    net.fullyConnect(net.layers[i], net.layers[i+1])

dataset = [
    ([1, 6],  [3.5, -5]),
    ([2, 4],  [3, -2]),
    ([3, 7],  [5, -4]),
    ([4, 8],  [6, -4]),
    ([5, 9],  [7, -4]),
    ([6, 12], [9, -6]),
    ([7, 10], [8.5, -3]),
    ([8, 16], [12, -8]),
    ([9, 11], [10, -2]),
    ([10, 13],[11.5, -3]),
    ([12, 15],[13.5, -3]),
    ([15, 20],[17.5, -5]),
    ([20, 30],[25, -10]),
]

net.train(dataset=dataset,maxEpochs=10000)
print(net.predict([2,12]))