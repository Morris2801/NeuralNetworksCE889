import math as m
import random as rd

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
        self.outputMax = 0
        self.outputMin = 0

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
    
    def exploreDataset(self, dataset):
        allIn = []
        allOut =[]
        for inputs, targets in dataset:
            allIn.extend(inputs)
            allOut.extend(targets)
        self.inputMax = max(allIn)
        self.inputMin = min(allIn)
        self.outputMax = max(allOut)
        self.outputMin = min(allOut)
    def normalize(self, value, minVal, maxVal):
        return(value-minVal) / (maxVal - minVal)
    def antinormalize(self,value,minVal,maxVal):
        return (value * (maxVal - minVal) + minVal)

    def train(self, dataset, maxEpochs): 
        self.exploreDataset(dataset)
        sseOld = 0
        sse = 0
        state = None 
        #fordebugg
        for epoch in range(maxEpochs):    
            totalSSE= 0
            rd.shuffle(dataset)
            for inputs, targets in dataset:
                for inputs, targets in dataset:
                    inputs = [self.normalize(x, self.inputMin, self.inputMax)for x in inputs]
                    targets = [self.normalize(y, self.outputMin, self.outputMax)for y in targets]
                        
                self.setInputs(inputs)
                self.forwardPss()
                outputDeltas =[]
                hiddenDeltas=[]
                sseOld=sse
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
                state = "Learning" if (sse < sseOld) else "Dumber" if (sse > sseOld) else "Hmmmm" 
            #print(f"Epoch {epoch} | TotalSSE:{totalSSE} | State {state}")
            if epoch % 1000 == 0: 
                print(f"Epoch {epoch}  | TOtal SSE={totalSSE}")
    
    def predict(self, inputs):
        inputs = [self.normalize(x, self.inputMin, self.inputMax) for x in inputs]

        self.setInputs(inputs)
        self.forwardPss()
        values = []
        for neuron in self.outputLayer.neurons:
            values.append(self.antinormalize(neuron.activation_val,self.outputMin,self.outputMax ))
        
        return values