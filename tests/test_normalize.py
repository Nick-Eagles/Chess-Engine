import sys
sys.path.append('../')

import Network
import network_helper
import Traversal
import input_handling

import random
import numpy as np

batchSize = 20
tol = 0.001

##########################################################
#   Initialize the network and generate examples
##########################################################

layers = []  # size of the input layer
messDef = "Define network architecture: how many hidden layers? "
messOnErr = "Not a valid number of layers"
cond = 'var >= 0 and var < 15'
numLayers = input_handling.getUserInput(messDef, messOnErr, 'int', cond)
for i in range(numLayers):
    print("Length of hidden layer", i, "?")
    layLen = int(input(": "))
    layers.append(layLen)
layers.append(1)    #   output of NN is a single value

slowNet = Network.Network(layers)

data = Traversal.full_high_R(slowNet)
data += Traversal.full_low_R(slowNet)

##########################################################
#   Test network_helper.normalize()
##########################################################

numBatches = int(len(data) / batchSize)
random.shuffle(data)
bs = batchSize
batches = []
for i in range(numBatches):
    temp = data[bs*i: bs*(i+1)]
    bGames = np.array([g[0].flatten() for g in temp]).T
    batches.append((bGames, np.array([g[1] for g in temp]).reshape((1,-1))))
                
for batch in batches:
    assert batch[0].shape[1] == bs
    z, zNorm, a = slowNet.ff_track(batch[0])

    for lay in zNorm:
        means = np.sum(lay, axis=1) / bs
        assert all(abs(means - np.full(lay.shape[0], 0.1)) < tol), means
        temp = lay - np.full(lay.shape, 0.1)
        var = np.sum(temp * temp, axis=1) / bs
        assert all(abs(var - np.ones(lay.shape[0])) < tol), var
