import numpy as np
from itertools import zip_longest

class Neuron:
    def __init__(self):
        self.input = []
        self.weights = ''
        self.bias = ''
        self.output = [] 
        self.tanoutput = ''

def tanh(x):
    return np.tanh(x)

def tanhDeriv(x):
    return 1.0 - np.tanh(x)**2

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

class NeuralNetwork:
    def __init__(self,inpnum,hnum,numbers=None):
        """
        Create a new NeuralNetwork object.
        'Numbers' represents the "brain" of the network i.e. stoers all the weights and biases.
        This is helpful for copying it into a new NN object. 
        If no 'numbers' value is passed, a new NN is created. If it is passed, a NN with the weights and biases of 'numbers' is created.
        """
        self.HiddenLayer = {}
        # initialise the object values, weights, and create the network
        if numbers == None:
            self.numbers = {} 
            self.numbers['Wx'] = np.random.normal(size=[inpnum,hnum]) # Input layer - Hidden layer weights
            self.numbers['Wh'] = np.random.normal(size=[hnum,1]) # Hidden layer - Output Layer weights
            self.numbers['hiddenbias'] = np.random.normal(size=[hnum,1])
            self.numbers['outputbias'] = np.random.normal()
        else:
            self.numbers = numbers

        Wx = self.numbers['Wx']
        Wh = self.numbers['Wh']
        HiddenBias = self.numbers['hiddenbias']
        OutputBias = self.numbers['outputbias']

        for n in range(hnum): 
            self.HiddenLayer[n] = Neuron()
            self.HiddenLayer[n].weights = Wx[:,n]
            self.HiddenLayer[n].bias = HiddenBias[n]

        self.OutputNode = Neuron()
        self.OutputNode.weights = Wh
        self.OutputNode.bias = OutputBias
    
    
    def feedforward(self,inputs,hnum):
        self.OutputNode.input = []
        for i in range(hnum): # iterate over all hidden layer neurons
            temp = []
            self.HiddenLayer[i].input = inputs
            
            self.HiddenLayer[i].output = np.dot(self.HiddenLayer[i].input,self.HiddenLayer[i].weights) + self.HiddenLayer[i].bias
            self.HiddenLayer[i].tanoutput = tanh(self.HiddenLayer[i].output)
            for output in self.HiddenLayer[i].tanoutput: #append all the outputs of hidden layer to input of outputnode
                temp.append(output)
            self.OutputNode.input.append(temp)
            
        self.OutputNode.input = np.array(self.OutputNode.input)
        self.OutputNode.output = np.dot(self.OutputNode.input.T, self.OutputNode.weights) + self.OutputNode.bias
        self.OutputNode.tanoutput = tanh(self.OutputNode.output) # Final output

        return self.OutputNode.tanoutput

