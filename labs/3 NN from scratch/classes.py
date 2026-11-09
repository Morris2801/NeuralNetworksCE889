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
            self.activation_val = self.activation_fn(acum)
        else:
            self.activation_val = acum
    
    def update_weights():
        pass #for now

##setup---------------------------
weights = [rd.uniform(-1,1) for _ in range(12)]
A, B, C, D, E, F, G, H, I, J, K, L = weights

#input
input1 = Neuron(2)
input2 = Neuron(4)

#hiddenlayer
hidden1 = Neuron(activation_fn=Neuron.activation_f_sigmoid)
hidden2 = Neuron(activation_fn=Neuron.activation_f_sigmoid)
hidden3 = Neuron(activation_fn=Neuron.activation_f_sigmoid)

#output
output1 = Neuron()
output2 = Neuron()

#connections
input1.connect(hidden1, A)
input1.connect(hidden2, B)
input1.connect(hidden3, C)
input2.connect(hidden1, D)
input2.connect(hidden2, E)
input2.connect(hidden3, F)
hidden1.connect(output1, G)
hidden1.connect(output2, H)
hidden2.connect(output1, I)
hidden2.connect(output2, J)
hidden3.connect(output1, K)
hidden3.connect(output2, L)

## activation ----------------------
for neuron in [hidden1, hidden2, hidden3]: 
    neuron.activate()
for neuron in [output1, output2]:
    neuron.activate()

##errors ------------------
target1 = 6
target2 = 4

error1 = target1- output1.activation_val
error2 = target2 - output2.activation_val

print("Output1 =", output1.activation_val)
print("Output2 =", output2.activation_val)
print("Error1 =", error1)
print("Error2 =", error2)
