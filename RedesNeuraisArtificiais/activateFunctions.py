import numpy as np

def stepFunction(sum):
    if sum >= 1:
        return 1
    return 0

def sigmoidFunction(sum):
    return 1 / (1 + np.exp(-sum))

def tahnFunction(sum):
    return (np.exp(sum) - np.exp(-sum)) / (np.exp(sum) + np.exp(-sum))

def reluFunction(sum):
    if sum >= 0:
        return sum
    return 0

def linearFunction(sum):
    return sum
    
def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()
    
testStep = stepFunction(2.1)
testSigmoid = sigmoidFunction(2.1)
testTahm = tahnFunction(2.1)
testRelu = reluFunction(2.1)
testLinear = linearFunction(2.1)

values = [5.0, 2.0, 1.3]
print(softmaxFunction(values))