import numpy as np
import random
from scipy.special import expit, logit
import os
from multiprocessing import Pool

import input_handling
import Game
import misc
import board_helper
import policy

#################################################################################
#   Utilities that are used by Network objects, but not quite suited
#   to be Network methods
#################################################################################

#   Given the input to a given layer and the learned parameters gamma and beta,
#   return the batch normalized input (which does not involve the application
#   of an activation function- this is done afterward)
def normalize(xBatch, gamma, beta, eps):
    assert xBatch.shape[0] > 0, "Tried to normalize an empty batch"
    assert len(xBatch.shape) == 2, "normalize() passed an illegitimate shaped array"
    assert len(gamma.shape) == 2 and gamma.shape[1] == 1, gamma.shape
    assert len(beta.shape) == 2 and beta.shape[1] == 1, beta.shape
    layLen = xBatch.shape[0]
    batchSize = xBatch.shape[1]
    
    mean = np.mean(xBatch, axis = 1).reshape((-1,1))
    dev = xBatch - np.dot(mean, np.ones((1, batchSize)))
    var_w_eps = np.add(np.mean(dev * dev, axis = 1), eps)
    var_w_eps = np.sqrt(var_w_eps)
    
    xNorm = np.add(np.array([gamma.flatten() * dev[:,i] / var_w_eps for i in range(batchSize)]).T, beta)
    assert xNorm.shape == (layLen, batchSize), xNorm.shape
    return xNorm

#   For a batch (2d np array w/ axis 0 being one example's activations), return a
#   tuple: (variance, sum of deviations, mean) as np arrays of expected dim
#def batchStats(xBatch):
#    mean = np.mean(xBatch, axis = 1)
#    dev = xBatch - np.dot(mean.reshape((-1,1)), np.ones((1, xBatch.shape[1])))
#    squareDev = dev * dev
#    return (np.mean(squareDev, axis = 1), squareDev, mean)

#   New method
def batchStats(xBatch, gamma, eps):
    mean = np.mean(xBatch, axis = 1).reshape((-1,1))
    dev = xBatch - np.dot(mean, np.ones((1, xBatch.shape[1])))
    squareDev = dev * dev
    var = np.add(np.mean(squareDev, axis = 1).reshape((-1,1)), eps)
    assert var.shape == (xBatch.shape[0], 1), var.shape

    #   These two terms are important for computing dC_dzNorm / dC_dz
    termA = squareDev / np.dot(var, np.ones((1, xBatch.shape[1])))
    termB = gamma / (xBatch.shape[1] * np.sqrt(var))
    termB = termB * np.ones((1, xBatch.shape[1]))
    assert termA.shape == xBatch.shape, termA.shape
    assert termB.shape == xBatch.shape, termB.shape
    
    return (termA, termB)

def toBatchChunks(data, bs, numCPUs):  
    numBatches = int(len(data) / bs)
    chunkSize = int(bs / numCPUs)
    remainder = bs % numCPUs
    
    #   Reformat data as tensors of correct dimension
    inTensor = np.array([x[0].flatten() for x in data]).T
    outTensor = np.array([x[1].flatten() for x in data]).T

    chunkedData = []
    for i in range(numBatches): 
        #   Subset all data to make batch
        bIns = inTensor[:, bs*i: bs*(i+1)]
        bOuts = outTensor[:, bs*i: bs*(i+1)]

        #   Break the batch into chunks
        bChunks = []
        c = 0
        for j in range(numCPUs):
            thisSize = chunkSize + (j < remainder)
            bChunks.append((bIns[:, c:c+thisSize], bOuts[:,c:c+thisSize]))
            c += thisSize

        #   Add the chunked batch to the list of chunk batches
        chunkedData.append(bChunks)

    #   Some dimensionality checks
    assert len(chunkedData) == numBatches
    assert len(chunkedData[0]) == numCPUs
    assert len(chunkedData[0][0]) <= bs + 1
    
    return chunkedData

def get_pop_stats(net, chunk):
    z = net.ff_track(chunk)[0]
    bs = chunk.shape[1]
    
    popMean, popVar = [], []
    for lay in range(len(z)):
        if lay in net.downSampLays:
            popMean.append('Undefined for projection layers!')
            popVar.append('Undefined for projection layers!')
        else:
            popMean.append(np.mean(z[lay], axis=1).reshape((-1,1)))
            popVar.append((bs * np.var(z[lay], axis=1) / (bs - 1)).reshape((-1,1)))

    return (popMean, popVar)


#   Intended to provide numerical stability when computing costs. y is the actual
#   reward for a given example, which can only be outside of [0,1] due to floating
#   point imprecision. 'tol' sets the threshold for acceptable level of deviation
#   from this range. Returns a value restricted to the range (0,1), where cutoff
#   determines how far in from the endpoints the function changes from linear to
#   asymptotic.
def stabilize(y, cutoff, tol):
    assert y < 1 + tol, y
    assert y > -1*tol, y
    if y > 1 - cutoff:
        return float(1 - cutoff * np.exp((1 - cutoff - y)/ cutoff))
    if y < cutoff:
        return float(cutoff * np.exp((y - cutoff)/ cutoff))
    return y

#   Given the Network "net" and training examples in typical tuple form "tData",
def net_activity(net, tData):
    p = input_handling.readConfig(0)
    bigBatch = np.array([g[0].flatten() for g in tData]).T
    z, zNorm, a = net.ff_track(bigBatch)

    print("Neuron activity by layer:")
    for i in range(len(net.layers) - 1):
        if i in net.downSampLays:
            print("(Linear downsample layer", i+1, "skipped)")
        else:
            num_dead = 0
            for j in range(a[i+1].shape[0]):
                num_dead += all(a[i+1][j,:] < 0)
                
            percDead = round(100 * num_dead / a[i+1].shape[0], 1)
            percActive = round(100 * np.mean(a[i+1] > 0), 1)

            print("Layer ", i+1, ": ", percActive, "% active; ", percDead, "% dead.", sep='')
    print("(Sigmoid output layer skipped)")

#   A helper function for Network.showGame. Returns the string that is output for
#   a game at its current state (the list of evaluations for whichever player is
#   to move). These are printed when running the program in mode >= 2.
def generateAnnLine(evalList, game):
    if game.whiteToMove:
        line = "#####  Move " + str(game.moveNum) + ":  #####\n\n-- White: --\n"
    else:
        line = "-- Black: --\n"
        
    for i, e in enumerate(evalList):
        line += str(i) + ". " + e[0] + ", overall: " + str(round(e[1] + e[2], 4))
        line += " | r = " + str(round(e[2], 4)) + " | NN eval = " + str(round(e[1], 4)) + "\n"
    line += "\n"

    return line

def bestGame(net):
    p = input_handling.readConfig(3)
    p.update(input_handling.readConfig(1))
    p['epsGreedy'] = 0
    p['epsSearch'] = 0

    game = Game.Game(quiet=False)

    pool = Pool()
    while (game.gameResult == 17):
        bestMove = policy.getBestMoveTreeEG(net, game, p, pool=pool)
        game.doMove(bestMove)
    pool.close()

    print(game.annotation)
    game.toPGN()
