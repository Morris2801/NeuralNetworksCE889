from neuron import Neuron, Layer, Network

##setup---------------------------
alpha = 0.1 # momentum
lambHidden = 0.8 # labmda
lambOutput = 0.75
biasHidden = 1
biasOutput = 1
eta = 0.6 #learning rate

#newtork
inputLayer = Layer(2)
hiddenLayer = Layer(4, activation_fn=Neuron.activation_f_sigmoid, lamb=lambHidden,bias= biasHidden)
outputLayer = Layer(2, bias=biasOutput)

net = Network(inputLayer, hiddenLayer, outputLayer, eta, alpha)
for i in range(len(net.layers)-1):
    net.fullyConnect(net.layers[i], net.layers[i+1])

net.train(inputs=[2,4], targets=[3,-2],maxEpochs=100)