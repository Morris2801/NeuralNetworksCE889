import math as m
import random as rd
import json
import os

targetDir = r"C:\Users\mauri\Coding\Python programs\Essex\Neural Networks CE889\labs\3 NN from scratch\weights"

##defs----------
class Neuron:
    def __init__(self, activation_val = 0, activation_fn = None, lamb = 1, bias = 0):
        self.activation_val = activation_val
        self.activation_fn = activation_fn
        self.lamb = lamb
        self.bias = bias
        self.previousdiffweight = {}
        self.previousdiffbias = 0
        self.connections = {"forward": {}, "backward":{}}
        
    def activation_f_sigmoid(input_val = 0, lamb = 1):
        return (1/(1 + m.exp(-lamb * input_val)))

    def connect(self, target, weight):
        self.connections["forward"][target]=weight
        target.connections["backward"][self]=weight    
    
    def activate(self):
        acum = self.bias
        for neuron, weight in self.connections["backward"].items():
            acum += neuron.activation_val * weight
        if self.activation_fn:
            self.activation_val = self.activation_fn(acum, self.lamb)
        else:
            self.activation_val = acum
    
    def update_weights(self, delta, eta,alpha):
        for neuron, weight in self.connections["backward"].items():
            dw = eta * delta * neuron.activation_val
            dw += alpha*self.previousdiffweight.get(neuron, 0)
            newWeight = weight + dw
            self.connections["backward"][neuron] = newWeight
            neuron.connections["forward"][self] = newWeight
            self.previousdiffweight[neuron] = dw
        db = eta*delta+alpha*self.previousdiffbias
        self.bias += db
        self.previousdiffbias = db

class Layer:
    def __init__(self, size, activation_fn = None, lamb = 1, bias =0):
        self.neurons=[]
        for _ in range(size):
            self.neurons.append(Neuron(activation_fn=activation_fn, lamb = lamb, bias = bias))
    def activate(self):
        for neuron in self.neurons: 
            neuron.activate()

class Network: 
    def __init__(self, inputLayer, hiddenLayer, outputLayer, eta, alpha):
        self.layers = [inputLayer, hiddenLayer, outputLayer]
        self.inputLayer=inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
        self.eta =eta
        self.alpha =alpha

        self.inputMax = 0
        self.inputMin = 0
        self.outputMaxs = []
        self.outputMins = []
        self.trainingSSELog = []

    def setInputs(self, inputs):
        for neuron, val in zip(self.inputLayer.neurons, inputs):
            neuron.activation_val = val
    def forwardPss(self):
        self.hiddenLayer.activate()
        self.outputLayer.activate()
    def fullyConnect(sef, sourceLayer, targetLayer):
        for source in sourceLayer.neurons:
            for target in targetLayer.neurons:
                source.connect(target, rd.uniform(-1,1)) #################################weights de -1 a 1
    
    def getRanges(self, dataset):
        allIn = []
        for inputs, targets in dataset:
            allIn.extend(inputs)
        self.inputMax = max(allIn)
        self.inputMin = min(allIn)
        numOutputs = len(dataset[0][1])
        self.outputMaxs =[]
        self.outputMins = []
        for i in range(numOutputs):
            values = []
            for _, targets in dataset: 
                values.append(targets[i])
            self.outputMins.append(min(values))
            self.outputMaxs.append(max(values))

    def normalize(self, value, minVal, maxVal):
        return(value-minVal) / (maxVal - minVal)
    def antinormalize(self,value,minVal,maxVal):
        return (value * (maxVal - minVal) + minVal)

    def getWeights(self):
        dataToSave = {
            "inputMin": self.inputMin, 
            "inputMax": self.inputMax,
            "outputMins": self.outputMins,
            "outputMaxs": self.outputMaxs,
            "layers": []
        }
        for layer in self.layers[1:]:
            layerData = []
            for neuron in layer.neurons:
                neuronData = {
                    "bias": neuron.bias, 
                    "weights": list(neuron.connections["backward"].values())
                }    
                layerData.append(neuronData)
            dataToSave["layers"].append(layerData)
        return dataToSave
    def saveWeights(self, f):
        os.makedirs(targetDir, exist_ok=True)
        filepath = os.path.join(targetDir, f)
        dataToSave = self.getWeights()
        with open(filepath, "w") as f: 
            json.dump(dataToSave, f, indent = 4)
    def loadWeights(self, f): 
        os.makedirs(targetDir, exist_ok=True)
        filepath = os.path.join(targetDir, f)
        with open(filepath, "r") as f: 
            data = json.load(f)
        self.inputMin = data["inputMin"]
        self.inputMax = data["inputMax"]
        self.outputMins = data["outputMins"]
        self.outputMaxs = data["outputMaxs"]
        layersData = data["layers"]
        for layerData, layer in zip(layersData, self.layers[1:]):
            for neuronData, neuron in zip(layerData, layer.neurons):
                neuron.bias = neuronData["bias"]
                for source, weight in zip(neuron.connections["backward"].keys(), neuronData["weights"]):
                    neuron.connections["backward"][source] = weight
                    source.connections["forward"][neuron] = weight
        print("Weights loaded")

    def train(self, dataset, maxEpochs): 
        self.getRanges(dataset)
        originalSSE = 0
        sse = 0
        state = None 
        #fordebugg
        oldtotalSSE = 0
        totalSSE = 0
        for epoch in range(maxEpochs):    
            oldtotalSSE = totalSSE
            totalSSE= 0
            rd.shuffle(dataset)
            for inputs, targets in dataset:
                inputs = [self.normalize(x, self.inputMin, self.inputMax)for x in inputs]
                targets = [self.normalize(y, self.outputMins[i], self.outputMaxs[i])for i, y in enumerate(targets)]            
                self.setInputs(inputs)
                self.forwardPss()
                outputDeltas =[]
                hiddenDeltas=[]
                sse = 0

                #outperrors
                for output, target in zip(self.outputLayer.neurons, targets):
                    error= target-output.activation_val
                    
                    sse += 0.5 *    (error**2)
                    outputDeltas.append(error)
                #hidderrors
                for hidden in self.hiddenLayer.neurons:
                    weightError = 0
                    for output, weight in hidden.connections["forward"].items():
                        index = (self.outputLayer.neurons.index(output))
                        weightError += outputDeltas[index] * weight
                    delta = hidden.lamb * hidden.activation_val * (1-hidden.activation_val) * weightError #############LINEARACTIV -> DERIV SIMPL
                    hiddenDeltas.append(delta)
                #updateweights
                for output, delta in zip(self.outputLayer.neurons, outputDeltas):
                    output.update_weights(delta, self.eta, self.alpha)
                for hidden, delta in zip(self.hiddenLayer.neurons, hiddenDeltas):
                    hidden.update_weights(delta, self.eta, self.alpha)
                totalSSE+=sse
            #print(f"Epoch {epoch} | TotalSSE:{totalSSE} | State {state}")
            if epoch == 0: 
                originalSSE = totalSSE
                self.trainingSSELog.append([epoch, originalSSE])
            if epoch % 1000 == 0: 
                state = "Learning" if (totalSSE< self.trainingSSELog[-1][1]) else "Dumber" if (totalSSE > self.trainingSSELog[-1][1]) else "Hmmmm"                 
                self.trainingSSELog.append([epoch, totalSSE])
                print(f"Epoch {epoch}  | TOtal SSE={totalSSE} | {state} by {totalSSE-self.trainingSSELog[-1][1]}")
            if (maxEpochs - epoch) == 1: 
                print(f"Tr. Results: SSE from {originalSSE} to {totalSSE}")
                print("---------------")
                self.saveWeights("XORTEST.json")
    
    def predict(self, inputs):
        inputs = [self.normalize(x, self.inputMin, self.inputMax) for x in inputs]

        self.setInputs(inputs)
        self.forwardPss()
        values = []
        for i, neuron in enumerate(self.outputLayer.neurons):
            values.append(self.antinormalize(neuron.activation_val,self.outputMins[i],self.outputMaxs[i] ))
        return values