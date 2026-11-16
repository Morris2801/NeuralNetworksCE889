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
    def setInputs(self, inputs):
        for neuron, val in zip(self.inputLayer.neurons, inputs):
            neuron.activation_val = val
    def forwardPss(self):
        self.hiddenLayer.activate()
        self.outputLayer.activate()
    def fullyConnect(sef, sourceLayer, targetLayer):
        for source in sourceLayer.neurons:
            for target in targetLayer.neurons:
                source.connect(target, rd.uniform(0,1)) #################################weights de 0 a 1
    def train(self, inputs, targets, maxEpochs): 
        self.setInputs(inputs)
        sseOld = 0
        sse = 0
        state = None 
        #fordebugg
        for epoch in range(maxEpochs):    
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
            state = "Learning" if (sse < sseOld) else "Dumber" if (sse > sseOld) else "Hmmmm" 
            print(f"Epoch {epoch} | SSE:{sse} | State {state}")
            if epoch > 1 and (sse <= 0.0000000001): 
                print("Early stopping criteria")
                break
