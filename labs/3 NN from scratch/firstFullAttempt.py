import random as rd
from neuron import Neuron

##setup---------------------------
alpha = 0.1 # momentum
lambHidden = 0.8 # labmda
lambOutput = 0.75
biasHidden = 1
biasOutput = 1
eta = 0.6 #learning rate
weights = [rd.uniform(-1,1) for _ in range(16)]
A, B, C, D, E , F, G, H , I , J , K, L, M, N, O , P= weights
sse = 0
sseOld = 0
flag = None

#input
input1 = Neuron(2)
input2 = Neuron(4)
target1 = (input1.activation_val + input2.activation_val )/ 2
target2 = -2

#hiddenlayer
hidden1 = Neuron(activation_fn=Neuron.activation_f_sigmoid, lamb=lambHidden, bias=biasHidden)
hidden2 = Neuron(activation_fn=Neuron.activation_f_sigmoid, lamb=lambHidden, bias=biasHidden)
hidden3 = Neuron(activation_fn=Neuron.activation_f_sigmoid, lamb=lambHidden, bias=biasHidden)
hidden4 = Neuron(activation_fn=Neuron.activation_f_sigmoid, lamb=lambHidden, bias=biasHidden)
hiddenlayer = [hidden1, hidden2, hidden3, hidden4]

#output
output1 = Neuron(lamb=lambOutput, bias = biasOutput)
output2 = Neuron(lamb=lambOutput ,bias = biasOutput)
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

## LOOP ----------------------
epochs = 100
for epoch in range(epochs):
    #forwardPass
    for neuron in hiddenlayer: 
        neuron.activate()
    for neuron in outputlayer:
        neuron.activate()

    #errors ------------------
    error1 = target1- output1.activation_val
    deltaO1 = error1
    error2 = target2 - output2.activation_val
    deltaO2 = error2
    sseOld = sse
    sse = 0.5 * (error1 **2 + error2**2)
    if epoch > 0: 
        flag = sse < sseOld
    else: 
        flag = None
    #backprop&weightupdating ----------------------------------
    delta_h1 = hidden1.lamb * hidden1.activation_val * (1 - hidden1.activation_val) * (deltaO1 * hidden1.connections["forward"][output1] + deltaO2 * hidden1.connections["forward"][output2])
    delta_h2 = hidden2.lamb * hidden2.activation_val * (1 - hidden2.activation_val) * (deltaO1 * hidden2.connections["forward"][output1] + deltaO2 * hidden2.connections["forward"][output2])
    delta_h3 = hidden3.lamb * hidden3.activation_val * (1 - hidden3.activation_val) * (deltaO1 * hidden3.connections["forward"][output1] + deltaO2 * hidden3.connections["forward"][output2])
    delta_h4 = hidden4.lamb * hidden4.activation_val * (1 - hidden4.activation_val) * (deltaO1 * hidden4.connections["forward"][output1] + deltaO2 * hidden4.connections["forward"][output2])

    output1.update_weights(deltaO1, eta, alpha)
    output2.update_weights(deltaO2, eta, alpha)
    hidden1.update_weights(delta_h1, eta, alpha)
    hidden2.update_weights(delta_h2, eta, alpha)
    hidden3.update_weights(delta_h3, eta, alpha)
    hidden4.update_weights(delta_h4, eta, alpha)

    #DEBUG
    print(f"Epoch {epoch}|| Output1:{output1.activation_val} | Output2:{output2.activation_val} || SSE:{sse} | Flag: {flag}")
    
    if sse == sseOld and flag == False: 
        print("Finished early")
        break
    
print("-----------")
print("Final Output1: ", output1.activation_val)
print("Final Output2: ", output2.activation_val)
print("Final SSE:", sse)