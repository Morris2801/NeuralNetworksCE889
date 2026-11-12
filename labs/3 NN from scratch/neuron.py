import math as m
import random as rd

##defs----------
class Neuron:
    def __init__(self, activation_val = 0, activation_fn = None):
        self.activation_val = activation_val
        self.activation_fn = activation_fn
        self.connections = {"forward": {}, "backward":{}}
        
    def activation_f_sigmoid(input_val = 0, lamb = 1):
        return (1/(1 + m.exp(-lamb * input_val)))

    def connect(self, target, weight):
        self.connections["forward"][target]=weight
        target.connections["backward"][self]=weight    
    
    def activate(self):
        acum = 0
        for neuron, weight in self.connections["backward"].items():
            acum += neuron.activation_val * weight
        if self.activation_fn:
            self.activation_val = self.activation_fn(acum, lamb)
        else:
            self.activation_val = acum
    
    def update_weights(self, delta, eta):
        for neuron, weight in self.connections["backward"].items():
            dw = eta * delta * neuron.activation_val
            newWeight = weight + dw
            self.connections["backward"][neuron] = newWeight
            neuron.connections["forward"][self] = newWeight


##setup---------------------------
alpha = 0.7 # learningrate
lamb = 0.8 # labmda
eta = 0.1 # momentum?
weights = [rd.uniform(-1,1) for _ in range(16)]
A, B, C, D, E , F, G, H , I , J , K, L, M, N, O , P= weights

#input
input1 = Neuron(2)
input2 = Neuron(4)

#hiddenlayer
hidden1 = Neuron(activation_fn=Neuron.activation_f_sigmoid)
hidden2 = Neuron(activation_fn=Neuron.activation_f_sigmoid)
hidden3 = Neuron(activation_fn=Neuron.activation_f_sigmoid)
hidden4 = Neuron(activation_fn=Neuron.activation_f_sigmoid)
hiddenlayer = [hidden1, hidden2, hidden3, hidden4]

#output
output1 = Neuron()
output2 = Neuron()
outputlayer = [output1, output2]

#connections
input1.connect(hidden1, A)
input1.connect(hidden2, B)
input1.connect(hidden3, C)
input1.connect(hidden4, D)
input2.connect(hidden1, E)
input2.connect(hidden2, F)
input2.connect(hidden3, G)
input2.connect(hidden4, H)
hidden1.connect(output1, I)
hidden2.connect(output1, J)
hidden3.connect(output1, K)
hidden4.connect(output1, L)
hidden1.connect(output2, M)
hidden2.connect(output2, N)
hidden3.connect(output2, O)
hidden4.connect(output2, P)

## activation ----------------------
for neuron in hiddenlayer: 
    neuron.activate()
for neuron in outputlayer:
    neuron.activate()

##errors ------------------

target1 = 6
target2 = 4

error1 = target1- output1.activation_val
deltaO1 = error1
error2 = target2 - output2.activation_val
deltaO2 = error2
sse = 0.5 * (error1 **2 + error2**2)
'''
print("Output1 =", output1.activation_val)
print("Output2 =", output2.activation_val)
print("Error1 =", error1)
print("Error2 =", error2)
print("SSE =aaaaaaaaaaaaaaaaaa", sse)
'''
# weightupdating ----------------------------------

delta_h1 = lamb * hidden1.activation_val * (1 - hidden1.activation_val) * (deltaO1 * hidden1.connections["forward"][output1] + deltaO2 * hidden1.connections["forward"][output2])
delta_h2 = lamb * hidden2.activation_val * (1 - hidden2.activation_val) * (deltaO1 * hidden2.connections["forward"][output1] + deltaO2 * hidden2.connections["forward"][output2])
delta_h3 = lamb * hidden3.activation_val * (1 - hidden3.activation_val) * (deltaO1 * hidden3.connections["forward"][output1] + deltaO2 * hidden3.connections["forward"][output2])
delta_h4 = lamb * hidden4.activation_val * (1 - hidden4.activation_val) * (deltaO1 * hidden4.connections["forward"][output1] + deltaO2 * hidden4.connections["forward"][output2])

#print("Before:", hidden1.connections["forward"][output1])
output1.update_weights(deltaO1, eta)
output2.update_weights(deltaO2, eta)

hidden1.update_weights(delta_h1, eta)
hidden2.update_weights(delta_h2, eta)
hidden3.update_weights(delta_h3, eta)
hidden4.update_weights(delta_h4, eta)

#print("After:", hidden1.connections["forward"][output1])
'''NOW UNUSED aftertest
#second forward pass 
for neuron in hiddenlayer: 
    neuron.activate()
for neuron in outputlayer: 
    neuron.activate()

newError1 = target1- output1.activation_val
newError2 = target2 - output2.activation_val
newSSE = 0.5 * (newError1 **2 + newError2**2)
print("After one training step")
print("output1: ", output1.activation_val)
print("output2: ", output2.activation_val)
print("Error1: ", newError1)
print("Error2: ", newError2)
print("SSE: ", newSSE)

if newSSE < sse: 
    print("Learning: SSE dec")
else:
    print("SSE up")
'''