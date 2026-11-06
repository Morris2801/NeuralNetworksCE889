import math as m

class Neuron:
    def __init__(self, activation_val, connections, weights):
        activation_val = 0
        connections = []
        weights = {}
        
    def activation_f_sigmoid(lamb = 1, input_val = 0):
        return (1/(1 + m.exp(lamb * input_val)))
        
    def update_weights():
        pass #for now

    def weight_multip(activation_val, weight):
        return activation_val * weight

n1 = Neuron(1)
n2 = Neuron(2)
n3 = Neuron(3)

layer1 = [n1]
layer2 = [n2]
layer3 = [n3]

n1.connections = [n2]
n2.connections = [n3]

network = [layer1, layer2, layer3]
