import math as m
import random as rd

##defs----------
class Neuron:
    def __init__(self, activation_val = 0):
        self.activation_val = activation_val
        self.connections = {"forward": {}, "backward":{}}
        
    def activation_f_sigmoid(lamb = 1, input_val = 0):
        return (1/(1 + m.exp(lamb * input_val)))
        
    def update_weights():
        pass #for now

    def weight_multip(activation_val, weight):
        return activation_val * weight

##setup------------
weights = [rd.uniform(-1,1) for i in range(12)]
A, B, C, D, E, F, G, H, I, J, K, L = weights
tempA, tempB, tempC, tempD, tempE, tempF, tempG, tempH, tempI, tempJ, tempK, tempL = [0]*12
#input
input1 = Neuron(2)
input2 = Neuron(4)
#hiddenlayer
hidden1 = Neuron()
hidden2 = Neuron()
hidden3 = Neuron()
#output
output1 = Neuron()
output2 = Neuron()

input1.connections = {
    "forward":{
        hidden1: A, 
        hidden2: B,
        hidden3: C,
    },
    "backward":{
    }
}
input2.connections = {
    "forward":{
        hidden1: D, 
        hidden2: E,
        hidden3: F,
    },
    "backward":{
    }
}
hidden1.connections = {
    "forward":{
        output1: G,
        output2: H,
    },
    "backward":{
        input1: tempA,
        input2: tempD,
    }
}
hidden2.connections = {
    "forward":{
        output1: I,
        output2: J,
    },
    "backward":{       
        input1: tempB,
        input2: tempE,
    }
}
hidden3.connections = {
    "forward":{
        output1: K,
        output2: L,
    },
    "backward":{       
        input1: tempC, 
        input2: tempF,
    }
}
output1.connections = {
    "forward": {
    },
    "backward":{
        hidden1: tempG, 
        hidden2: tempI,
    }
}
output2.connections = {
    "forward":{
    },
    "backward":{
        hidden2: tempJ,
        hidden3: tempL
    }
}

##hidden activation fowrard
hidden1.activation_val = Neuron.activation_f_sigmoid(
    input1.activation_val * A + input2.activation_val * D
)
hidden2.activation_val = Neuron.activation_f_sigmoid(
    input1.activation_val * B + input2.activation_val * E
)
hidden3.activation_val = Neuron.activation_f_sigmoid(
    input1.activation_val * C + input2.activation_val * F
)
##output activation forward (linear)
output1.activation_val = (
    hidden1.activation_val *G + hidden2.activation_val * I + hidden3.activation_val * K
)
output2.activation_val = (
    hidden1.activation_val * H + hidden2.activation_val * J + hidden3.activation_val *L
)

target1 =6
target2= 4

error1 = target1 - output1.activation_val
error2 = target2 - output2.activation_val

print("Output1:")
print(output1.activation_val)
print("Output2:")
print(output2.activation_val)

print("Error1:")
print(error1)
print("Error2:")
print(error2)
