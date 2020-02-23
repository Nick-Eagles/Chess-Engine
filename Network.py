import numpy as np
from scipy.special import expit, logit
import copy
import json
import random
import csv
import os
from multiprocessing import Pool
import time

import Game
import board_helper
import network_helper
import input_handling
import Traversal
import Move
import file_IO

class Network:
    def __init__(self, layers, blockWidth, blocksPerGroup, weights=[], biases=[], beta=[], gamma=[], popMean=[], popVar=[], tCosts=[], vCosts=[], age=0, experience=0, certainty=0, certaintyRate=0):
        #   Standard deviation for normalized activations of the last layer in residual blocks
        self.alpha = 0.1

        initial_bias = 0.0
        var_scale = 1.0
        
        #   Layer numbers for those layers receiving activations from earlier layers, and those used in later layers, respectively
        self.resInputs = [i for i in range(len(layers)) if i > blockWidth and (i + int((i - 2) / (blockWidth * blocksPerGroup + 1))) % blockWidth == 1]
        self.resOutputs = [i - (blockWidth + 1) for i in self.resInputs]
        
        #   Weights and biases "beta"
        if len(weights) == 0:
            temp = [784] + layers
            
            self.weights = []
            for i in range(len(layers)):
                #   Scale weights so that stddev of every layer's input is 1
                if i - 1 in self.resInputs:
                    scalar = np.sqrt(temp[i] * (self.alpha**2 + 1)) / var_scale
                else:
                    scalar = np.sqrt(temp[i]) / var_scale
                    
                self.weights.append(np.random.randn(temp[i+1], temp[i]) / scalar)
            
        else:
            self.weights = weights
        if len(biases) == 0:
            self.biases = [np.full((x, 1), initial_bias) for x in layers[:-1]]
        else:
            self.biases = biases

        #   Ideal activation means "beta" for batch norm
        if len(beta) == 0:
            self.beta = [np.zeros((x, 1)) for x in layers]
        else:
            self.beta = beta
            
        #   Ideal activation stddevs "gamma" for batch norm
        if len(gamma) == 0:
            self.gamma = []
            for i in range(len(layers)):
                if i + 1 in self.resInputs:
                    self.gamma.append(np.full((layers[i], 1), self.alpha * var_scale))
                else:
                    self.gamma.append(np.full((layers[i], 1), var_scale))
        else:
            self.gamma = gamma

        #   Population mean, variance, and stddev for inputs to each layer
        if len(popMean) == 0:
            self.popMean = [np.zeros((x, 1)) for x in layers]
        else:
            self.popMean = popMean
        if len(popVar) == 0:
            self.popVar = [np.ones((x, 1)) for x in layers]
        else:
            self.popVar = popVar
            
        self.eps = 0.000001
        self.popDev = [np.sqrt(np.add(lay, self.eps)) for lay in self.popVar]

        #   List of lengths of hidden layers, in forward order
        self.layers = layers
        self.blockWidth = blockWidth
        self.blocksPerGroup = blocksPerGroup

        #   For momentum term in SGD
        self.last_dC_dw = [np.zeros(lay.shape) for lay in self.weights]
        self.last_dC_dbias = [np.zeros(lay.shape) for lay in self.biases]
        self.last_dC_dbeta = [np.zeros(lay.shape) for lay in self.beta]
        self.last_dC_dg = [np.zeros(lay.shape) for lay in self.gamma]

        #   For analysis of cost vs. epoch after training
        self.tCosts = tCosts
        self.vCosts = vCosts

        self.age = age  # number of training steps
        self.experience = experience # number of unique training examples seen
        self.certainty = certainty # see README- a measure of how much observed recent events match expected ones
        self.certaintyRate = certaintyRate

    def copy(self):
        weights = [lay.copy() for lay in self.weights]
        biases = [lay.copy() for lay in self.biases]
        beta = [lay.copy() for lay in self.beta]
        gamma = [lay.copy() for lay in self.gamma]
        popMean = [lay.copy() for lay in self.popMean]
        popVar = [lay.copy() for lay in self.popVar]
        
        return Network(self.layers, self.blockWidth, self.blocksPerGroup, weights, biases, beta, gamma, popMean, popVar, self.tCosts, self.vCosts, self.age, self.experience, self.certainty, self.certaintyRate)
        
    #   Prints an annotated game of the neural network playing itself
    def showGame(self, verbose=True):
        p = input_handling.readConfig(1)
   
        game = Game.Game(quiet=False)

        if verbose:
            fullAnnotation = []
            rExpected = []
            rActual = []
            while (game.gameResult == 17):
                legalMoves = board_helper.getLegalMoves(game)
                fullEvals = []
                for m in range(len(legalMoves)):
                    rTuple = game.getReward(legalMoves[m], p['mateReward'])
                    NN_eval = float(logit(self.feedForward(rTuple[1])))
                    
                    #   Append (move name, immediate reward, NN eval for future reward)
                    fullEvals.append((legalMoves[m].getMoveName(game.board),
                                      float(logit(self.feedForward(rTuple[1]))),
                                      rTuple[0]))
                    val = fullEvals[-1][1] + fullEvals[-1][2]

                    #   See if this move is the best so far (this method is kept to ensure identical results
                    #   with method when verbose=False, though the following sort step would be slightly better)
                    if m == 0 or (game.whiteToMove and val > bestVal) or (not game.whiteToMove and val < bestVal):
                        bestVal = val
                        bestMove = legalMoves[m]

                #   Sort by sum of immediate and future expected rewards, and reverse order for black
                fullEvals.sort(key = lambda x: (-2 * game.whiteToMove + 1) * (x[1] + x[2]))
                fullAnnotation.append(network_helper.generateAnnLine(fullEvals, game))

                if game.gameResult == 17:
                    rExpected.append(fullEvals[0][1])
                if game.moveNum > 1:
                    rActual.append(fullEvals[0][2])
                
                game.doMove(bestMove)
                    
            for line in fullAnnotation:
                print(line)

            #   A metric I think makes intuitive sense for measuring how well expected reward aligns with
            #   actual reward. Similar to correlation, but accounts for differences in magnitude. -1 is perfect
            #   antisymmetry (magnitudes of reward are all equal and opposite); 1 is a perfect match, and 0
            #   roughly implies no meaningful association.
            synchronicity = 0
            for i in range(len(rActual)-p['rDepth']):
                x, y = rExpected[i], sum(rActual[i:(i+1+p['rDepth'])])
                synchronicity += x * y / max(abs(x), abs(y))**2
            synchronicity /= (len(rExpected)-1)

            print("Synchronicity:", synchronicity)

        else:
            #   Start and finish game
            while (game.gameResult == 17):
                #   Find and do the best legal move (the move maximizing the sum of immediate reward
                #   and expected cumulative reward from the resulting position)
                legalMoves = board_helper.getLegalMoves(game)
                for m in range(len(legalMoves)):
                    rTuple = game.getReward(legalMoves[m], p['mateReward'])
                    val = rTuple[0] + float(logit(self.feedForward(rTuple[1])))               
                    if m == 0 or (game.whiteToMove and val > bestVal) or (not game.whiteToMove and val < bestVal):
                        bestVal = val
                        bestMove = legalMoves[m]

                game.doMove(bestMove)

            print(game.annotation)  # Normal, non-verbose annotation
            
        print(game.gameResult, game.gameResultStr)
        print("Writing game to pgn file:")
        game.toPGN()
        print("Done.")

    #   Inference using fixed parameters; inputs are normalized by the expected population's
    #   mean and variance. "aNorm" is the input for a single example.
    def feedForward(self, aNorm):
        assert len(aNorm.shape) == 2 and aNorm.shape[1] == 1, aNorm.shape
        for lay in range(len(self.layers) - 1):
            #   Add identity for layers starting a residual block
            if lay in self.resInputs:
                aNorm += aLastBlock
                
            #   Nonlinearity (leaky ReLU)
            a = leakyReLU(self.weights[lay] @ aNorm + self.biases[lay])

            #   Batch normalization
            aNorm = (a - self.popMean[lay]) * self.gamma[lay] / self.popDev[lay] + self.beta[lay]
            #print("aNorm shape for layer ", lay+1, ":", aNorm.shape)
            #print(self.weights[lay].shape, a.shape, self.popMean[lay].shape, self.popDev[lay].shape)
            
            #   The start of a residual block has activations used [self.blockWidth] layers later
            if lay in self.resOutputs:
                aLastBlock = aNorm.copy()

        #   Batch normalization THEN nonlinearity (sigmoid) for the last layer
        #print((self.weights[-1] @ (aNorm + aLastBlock)).shape)
        aNorm = expit((self.weights[-1] @ (aNorm + aLastBlock) - self.popMean[-1]) * self.gamma[-1] / self.popDev[-1] + self.beta[-1])
        assert aNorm.shape == (47,1)
        return aNorm

    #   Inference for training on all members of a batch simultaneously
    def ff_track(self, aInput):
        assert len(aInput.shape) == 2 and aInput.shape[0] == 784, aInput.shape
        z, a = [], []
        aNorm = [aInput]
        for lay in range(len(self.layers) - 1):
            #   Add identity for layers starting a residual block. "a" is the raw activation
            #   (before normalization)
            if lay in self.resInputs:
                #print("Added activation's shape: ", (aNorm[lay] + aNorm[lay - self.blockWidth]).shape)
                #print("Biases shape: ", self.biases[lay].shape)
                assert aNorm[lay].shape == aNorm[lay - self.blockWidth].shape, str(aNorm[lay].shape) + str(aNorm[lay - self.blockWidth].shape)
                z.append(self.weights[lay] @ (aNorm[lay] + aNorm[lay - self.blockWidth]) + self.biases[lay])
            else:
                z.append(self.weights[lay] @ aNorm[lay] + self.biases[lay])
            a.append(leakyReLU(z[-1]))

            #   Batch normalized activation to feed to next layer
            aNorm.append(network_helper.normalize(a[lay], self.gamma[lay], self.beta[lay], self.eps))

        #   Note that batch norm is applied before sigmoid, and that therefore:
        #       1. biases are not used (aside for beta)
        #       2. a and aNorm have slightly different meanings, particularly for backprop
        z.append(self.weights[-1] @ (aNorm[-1] + aNorm[-1 - self.blockWidth]))
        #   Also note that a shortcut is used for the backprop step at the output layer, and a is not needed
        aNorm.append(expit(network_helper.normalize(z[-1], self.gamma[-1], self.beta[-1], self.eps)))
        assert aNorm[-1].shape == (47, aInput.shape[1]), aNorm[-1].shape
        
        return (z, a, aNorm)

    #   Trains on n = "numGames" games. Each game provides n = movecap training
    #   examples, so a mini-batch contains (movecap * batchSize) examples. SGD
    #   is performed otherwise as expected, and one epoch is completed.
    #
    #   nu:     as per convention, the learning rate of the NN.
    #   batchSize: the number of games per batch for SGD
    def train(self, games, vGames, p):
        print("----------------------")
        print("Beginning training...")
        print("----------------------")

        nu = p['nu']
        bs = p['batchSize']
        epochs = p['epochs']
        numCPUs = os.cpu_count()
        numBatches = int(len(games) / bs)
        
        #   Training via SGD
        if p['mode'] >= 2:
            start_time = time.time()
        pool = Pool()

        #   Make sure the examples are in random order for computation of cost
        #   and population statistics in batches
        #random.shuffle(games)
        random.shuffle(vGames)
        
        for epoch in range(epochs):
            #   Divide data into batches, which each are divided into chunks-
            #   computation of the gradients will be parallelized by chunk
            random.shuffle(games)
            chunkedGames = network_helper.toBatchChunks(games, bs, numCPUs)

            #   Baseline calculation of cost on training, validation data
            if epoch == 0:
                self.tCosts.append(self.totalCost(games, p))
                self.vCosts.append(self.totalCost(vGames, p))
            
            for i in range(numBatches):
                ###############################################################
                #   Submit the multicore 'job' and average returned gradients
                ###############################################################
                inList = [(self, chunk, p) for chunk in chunkedGames[i]]
                gradients_list = pool.starmap_async(train_thread, inList).get()

                #   Add up gradients by batch and parameter
                gradient = gradients_list[0]
                for j in range(1,len(gradients_list)):
                    #   Loop through each parameter (eg. weights, beta, etc)
                    for k in range(len(gradients_list[0])):
                        gradient[k] += gradients_list[j][k]

                ###############################################################
                #   Update parameters via SGD with momentum
                ###############################################################

                if i == 0 and p['mode'] >= 2:
                    for lay in range(len(self.weights)):
                        print("----------------------------------")
                        print("  Layer ", lay, ":", sep="")
                        print("----------------------------------")
                        temp0 = round(nu * np.linalg.norm(gradient[0][lay]), 3)
                        temp1 = round(np.linalg.norm(self.weights[lay]), 2)
                        print("Magnitude of weight partial (weight value): ", temp0, " (", temp1, ")", sep="")
                        temp0 = round(nu * np.linalg.norm(gradient[1][lay]), 3)
                        temp1 = round(np.linalg.norm(self.beta[lay]), 2)
                        print("Magnitude of beta partial (beta value): ", temp0, " (", temp1, ")", sep="")
                        temp0 = round(nu * np.linalg.norm(gradient[2][lay]), 3)
                        temp1 = round(np.linalg.norm(self.gamma[lay]), 2)
                        print("Magnitude of gamma partial (gamma value): ", temp0, " (", temp1, ")", sep="")
                        if lay < len(self.weights) - 1:
                            temp0 = round(nu * np.linalg.norm(gradient[3][lay]), 3)
                            temp1 = round(np.linalg.norm(self.biases[lay]), 2)
                            print("Magnitude of bias partial (bias value): ", temp0, " (", temp1, ")", sep="")
                            
                for lay in range(len(self.weights)):
                    self.last_dC_dw[lay] = gradient[0][lay] + p['mom'] * self.last_dC_dw[lay]
                    self.weights[lay] -= nu * self.last_dC_dw[lay]

                    self.last_dC_dbeta[lay] = gradient[1][lay].reshape((-1,1)) + p['mom'] * self.last_dC_dbeta[lay]
                    self.beta[lay] -= p['batchNormScale'] * nu * self.last_dC_dbeta[lay]

                    self.last_dC_dg[lay] = gradient[2][lay].reshape((-1,1)) + p['mom'] * self.last_dC_dg[lay]
                    self.gamma[lay] -= p['batchNormScale'] * nu * self.last_dC_dg[lay]

                    if lay < len(self.weights) - 1:
                        self.last_dC_dbias[lay] = gradient[3][lay].reshape((-1,1)) + p['mom'] * self.last_dC_dbias[lay]
                        self.biases[lay] -= nu * self.last_dC_dbias[lay]
                
            #   Approximate loss using batch statistics
            if p['mode'] >= 1 and epoch < epochs - 1:
                self.tCosts.append(self.totalCost(games, p))
                self.vCosts.append(self.totalCost(vGames, p))
            elif epoch == epochs - 1:
                self.setPopStats(games + vGames, p)
                self.tCosts.append(self.totalCost(games, p))
                self.vCosts.append(self.totalCost(vGames, p))
                
            print("Finished epoch ", epoch+1, ".", sep="")

        pool.close()
        if p['mode'] >= 2:
            elapsed = time.time() - start_time
            speed = elapsed / (epochs * numBatches * bs)
            print("Done training in ", round(elapsed, 2), " seconds (", round(speed, 6), " seconds per training example per epoch).", sep="")
        else:
            print("Done training.")
            
        self.age += numBatches * epochs
        
        #   Write cost analytics to file
        if p['mode'] >= 1:
            self.costToCSV(epochs)
        else:
            self.costToCSV(1)

    #   Backprop is performed at once on the entire batch, where:
    #       -z, a, and aNorm are lists of 2d np arrays, with index of axis 1 in each array
    #        corresponding to the training example, and each list element referring
    #        to a NN layer
    #       -y is a 2d np array following this formatting style
    def backprop(self, z, a, aNorm, y, p):
        #   Sanity checks on number of layers
        assert len(z) == len(self.layers)
        assert len(a) == len(self.layers) - 1
        assert len(aNorm) == len(self.layers) + 1

        assert aNorm[0].shape == (784, y.shape[1]), aNorm[0].shape
        assert aNorm[-1].shape == y.shape, aNorm[-1].shape

        if p['mode'] >= 2:
            bs = y.shape[1]
            assert aNorm[0].shape == (784, bs), aNorm[0].shape
            assert aNorm[-1].shape == y.shape, aNorm[-1].shape
            for i in range(len(self.layers)):
                assert z[i].shape == (self.layers[i], bs), z[i].shape
                assert aNorm[i+1].shape == (self.layers[i], bs), aNorm[i+1].shape
                if i < len(self.layers) - 1:
                    assert a[i].shape == (self.layers[i], bs), a[i].shape
        
        nu = p['nu']
        mom = p['mom']
        
        #   Initialize partials
        dC_dw = [np.zeros(self.weights[i].shape) for i in range(len(self.weights))]
        dC_dg = [np.zeros(self.gamma[i].shape) for i in range(len(self.gamma))]
        dC_dbeta = [np.zeros(self.beta[i].shape) for i in range(len(self.beta))]
        dC_dbias = [np.zeros(self.biases[i].shape) for i in range(len(self.biases))]
        dC_dz = [np.zeros(z[i].shape) for i in range(len(z))]

        batchSize = y.shape[1]
        assert batchSize > 0, "Attempted to do backprop on a batch of size 0"
        for i in range(len(z)):
            ###########################################################################
            #   First compute up to dC_daNorm, the partial w/ respect to the batch
            #   normalized activation, for the current layer
            ###########################################################################
            if i == 0:  # output layer
                #   This is actually dC_dzNorm, but renamed for consistent batchNorm calculation
                dC_daNorm = aNorm[-1] - y   # assumes cross-entropy loss and sigmoid activation; 2D
            elif len(z) - 1 - i in self.resOutputs:
                #   input layers to residual blocks
                dC_daNorm = np.dot(self.weights[-1*i].T, dC_dz[-1*i]) + np.dot(self.weights[self.blockWidth - i].T, dC_dz[self.blockWidth - i])
            else:
                dC_daNorm = np.dot(self.weights[-1*i].T, dC_dz[-1*i])

            #print("dC_daNorm.shape:", dC_daNorm.shape)
            #   Sum up partials w/ respect to gamma and beta over the batch, and divide by batchSize
            dC_dg[-1-i] = np.sum(dC_daNorm * (aNorm[-1-i] - self.beta[-1-i]) \
                                 / self.gamma[-1-i], axis = 1) / p['batchSize']
            #print("dC_dg[-1-1].shape:", dC_dg[-1-i].shape)
            dC_dbeta[-1-i] = np.sum(dC_daNorm, axis=1) / p['batchSize']
            #print("dC_dbeta[-1-1].shape:", dC_db[-1-i].shape)
                
            ###########################################################################
            #   Compute daNorm/da (account for batch normalization in gradient flow)
            ###########################################################################
            if i == 0:
                bStats = network_helper.batchStats(z[-1])
            else:
                bStats = network_helper.batchStats(a[-1*i])

            firstTerm = self.gamma[-1-i].flatten() /(batchSize * np.sqrt(bStats[0] + self.eps)) # 1D
            firstTerm = np.dot(firstTerm.reshape(-1,1), np.ones((1, batchSize)))    # 2D
            secTerm = batchSize - 1 - bStats[1] / np.dot(bStats[0].reshape(-1,1), np.ones((1, batchSize))) # 2D

            #   The term in parenthesis is daNorm/da (2D)
            dC_da = dC_daNorm * (firstTerm * secTerm)
            if i == 0:
                #assert dC_da.shape == dC_dz[-1-i].shape, dC_da.shape
                dC_dz[-1-i] = dC_da.copy() # recall we named zNorm 'aNorm' and z 'a', for the output layer
            else:
                #assert (dC_da * d_dx_leakyReLU(z[-1-i])).shape == dC_dz[-1-i].shape, (dC_da * d_dx_leakyReLU(z[-1-i])).shape
                dC_dz[-1-i] = dC_da * d_dx_leakyReLU(z[-1-i])

            ###########################################################################
            #   Compute the final partials dC_dw and dC_dbias, averaging over the batch
            ###########################################################################
            
            #   Sum up the partials w/ respect to the weights one training example at a time
            for j in range(batchSize):
                #   "residual input" layers take activations from a few layers before, affecting partials
                if len(z) - 1 - i in self.resInputs:
                    actual_a = (aNorm[-2-i][:,j] + aNorm[-2-i-self.blockWidth][:,j]).reshape((1,-1))
                else:
                    actual_a = aNorm[-2-i][:,j].reshape((1,-1))

                dC_dw[-1-i] = dC_dw[-1-i] + np.dot(dC_dz[-1-i][:,j].reshape((-1,1)), actual_a)
                
            dC_dw[-1-i] = dC_dw[-1-i] / p['batchSize'] + (p['weightDec'] / os.cpu_count()) * self.weights[-1-i]

            #   Sum up partials w/ respect to biases (except for the last layer, which
            #   doesn't have biases, because batch norm occurs before the nonlinearity)
            if i > 0:
                dC_dbias[-1*i] = (np.sum(dC_dz[-1-i], axis=1) / p['batchSize']).reshape((-1,1))

        return [dC_dw, dC_dbeta, dC_dg, dC_dbias]

    #   Given the entire set of training examples, returns the average cost per example.
    def totalCost(self, games, p):
        numCPUs = os.cpu_count()
        chunkSize = int(p['batchSize'] / numCPUs)
        numGames = min(p['costLimit'], len(games))
        remainder = numGames % chunkSize

        c = 0
        chunks = []
        for i in range(int(numGames / chunkSize)):
            thisSize = chunkSize + (i < remainder)
            chunks.append(games[c:c+thisSize])
            c += thisSize

        pool = Pool()
        costList = pool.map_async(self.batchLoss, chunks).get()
        pool.close()

        return sum(costList) / len(costList)

    #   Return the average loss for examples in the typical 'list of tuples' format:
    #   compute in chunks, as training is done (thus not relying on pop stats)
    def batchLoss(self, data):
        inBatch = np.array([x[0].flatten() for x in data]).T
        labels = np.array([x[1] for x in data]).reshape(1,-1)

        outBatch = self.ff_track(inBatch)[2][-1]
        costs = -1 * (labels * np.log(outBatch) + (1 - labels) * np.log(1 - outBatch))

        return float(np.sum(costs)) / len(data)

    #   Return the average loss for examples in the typical 'list of tuples' format:
    #   rely on population statistics, thus giving the highest quality estimate of
    #   loss (data was generated using pop stats, not grouped in batches)
    def individualLoss(self, data):
        a = np.array([self.feedForward(x[0]) for x in data])
        labels = np.array([x[1] for x in data])

        return -1 * float(np.mean(labels * np.log(a) + (1 - labels) * np.log(1 - a)))
        

    #   Writes the info about costs vs. epoch, that was saved during training,
    #   to a .csv file. This is designed to produce a temporary file that an
    #   R script can read and generate informative plots from.
    def costToCSV(self, epochs):
        #   Generate a a list of "epochNum, cost, cost type label" sets for each
        #   cost type recorded: this format is designed to take advantage of R's
        #   "ggplot2" package
        append = len(self.tCosts) > epochs + 1
        if append:
            costData = []
        else:
            costData = [["epochNum", "cost", "costType", "isStart"]]

        costTypes = [self.tCosts, self.vCosts]
        costLabels = ["t_cost", "v_cost"]

        numEpisodes = int(len(self.tCosts) / (epochs + 1))
        for epoch in range(epochs + 1):
            costIndex = 0
            for costType in costTypes:
                e = len(self.tCosts) - 1 - epochs + epoch
                costData.append([e - numEpisodes + 1, costType[e], costLabels[costIndex], int(epoch == 0)])
                costIndex += 1

        #   Open "costs.csv" and write the cost data
        filename = "visualization/costs.csv"
        try:
            #   Want to combine episodes into one big table
            if append:
                costFile = open(filename, 'a')
            else:
                costFile = open(filename, 'w')

            #   The actual writing of data
            with costFile:
                writer = csv.writer(costFile)
                writer.writerows(costData)
            print("Writing to costs.csv complete.")
        except:
            print("Encountered an error while opening or handling file 'costs.csv'")
        finally:
            costFile.close()

    #   Takes a list of tuples representing training data, and sets population mean, variance
    #   and standard deviation for the network (used for feedForward())
    def setPopStats(self, pop, p):
        #   "unbiased" estimate of population statistics: compute mean and variance
        #   in batches and average over these, as suggested in https://arxiv.org/pdf/1502.03167.pdf
        bs = p['batchSize']
        numCPUs = os.cpu_count()
        numBatches = int(len(pop) / bs)

        #   Compute stats in chunks (so pop stats are computed in a consistent way as stats
        #   are used during training)- this is equivalent to using a batch size of bs/numCPUs
        chunkedData = network_helper.toBatchChunks(pop, bs, numCPUs)

        pool = Pool()

        popMean = [np.zeros((lay, 1)) for lay in self.layers]
        popVar = [np.zeros((lay, 1)) for lay in self.layers]
        for i in range(numBatches):
            #   This is a list of tuples of lists: popStatList[chunkNum[statType[layerNum]]] is
            #   a numpy array representing the mean or variance of the activations for one layer
            #   from feeding forward one chunk
            inList = [(self, x[0]) for x in chunkedData[i]]
            popStatList = pool.starmap_async(network_helper.get_pop_stats, inList).get()

            for chunk in popStatList:
                for lay in range(len(self.layers)):
                    popMean[lay] += chunk[0][lay] / (numBatches * numCPUs)
                    popVar[lay] += chunk[1][lay] / (numBatches * numCPUs)

        pool.close()

        if p['mode'] >= 2:
            concDirMean = 0
            diffMagMean = 0
            concDirVar = 0
            diffMagVar = 0
        for i in range(len(self.popMean)):
            if p['mode'] >= 2:
                oldNorm = float(np.linalg.norm(self.popMean[i]))
                newNorm = float(np.linalg.norm(popMean[i]))
                concDirMean += abs(float(np.dot(self.popMean[i].T, popMean[i]))) / (oldNorm * newNorm)
                diffMagMean += abs(float(newNorm - oldNorm)) / (oldNorm + newNorm)

                oldNorm = float(np.linalg.norm(self.popVar[i]))
                newNorm = float(np.linalg.norm(popVar[i]))
                concDirVar += abs(float(np.dot(self.popVar[i].T, popVar[i]))) / (oldNorm * newNorm)
                diffMagVar += abs(float(newNorm - oldNorm)) / (oldNorm + newNorm)
                
            self.popMean[i] = p['popPersist'] * self.popMean[i] + (1 - p['popPersist']) * popMean[i]
            self.popVar[i] = p['popPersist'] * self.popVar[i] + (1 - p['popPersist']) * popVar[i]
            self.popDev[i] = np.sqrt(np.add(self.popVar[i], self.eps))
        if p['mode'] >= 2:
            print('Average normalized dot product of new and old pop stats:')
            print('  Means:', round(concDirMean / len(self.popMean), 5))
            print('  Vars:', round(concDirVar / len(self.popMean), 5))

            print('Average percent difference of new and old pop stats:')
            print('  Means:', round(100 * diffMagMean / len(self.popMean), 3))
            print('  Vars:', round(100 * diffMagVar / len(self.popMean), 3))

    def print(self):
        print('Layers:', self.layers)
        print('First 5 weights for each layer:')
        for lay in self.weights:
            print(np.round_(lay[0][:5], 4))
        print('First 5 biases:')
        for lay in self.beta:
            print(np.round_(lay[:5], 4).T)
        print('First 5 input means:')
        for lay in self.popMean:
            print(np.round_(lay[:5], 4).T)
        print('First 5 input vars:')
        for lay in self.popVar:
            print(np.round_(lay[:5], 4).T)
        print('Number of training steps total:', self.age)
        print('Unique examples seen: ~', self.experience, sep="")
        print('Certainty:', round(self.certainty, 4))
        print('Rate of certainty change:', round(self.certaintyRate, 5))

    def save(self, tBuffer, vBuffer, filename):       
        data = {"layers": self.layers,
                "weights": [w.tolist() for w in self.weights],
                "beta": [b.tolist() for b in self.beta],
                "gamma": [g.tolist() for g in self.gamma],
                "popMean": [m.tolist() for m in self.popMean],
                "popVar": [v.tolist() for v in self.popVar],
                "age": self.age,
                "experience": self.experience,
                "certainty": self.certainty,
                "certaintyRate": self.certaintyRate}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

        #   Write each sub-buffer to separate file
        for i in range(4):
            file_IO.writeGames(tBuffer[i], 'data/tBuffer' + str(i) + '.csv', True, False)
            file_IO.writeGames(vBuffer[i], 'data/vBuffer' + str(i) + '.csv', True, False)

def train_thread(net, batch, p):  
    z, zNorm, a = net.ff_track(batch[0])
    
    return net.backprop(np.array(z), np.array(zNorm), np.array(a), batch[1], p)
      
def load(filename, lazy=False):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    
    net = Network(data["layers"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.beta = [np.array(b) for b in data["beta"]]
    net.gamma = [np.array(g) for g in data["gamma"]]
    net.popMean = [np.array(m) for m in data["popMean"]]
    net.popVar = [np.array(v) for v in data["popVar"]]
    net.popDev = [np.sqrt(np.add(m, net.eps)) for m in net.popVar]
    net.age = data["age"]
    net.experience = data["experience"]
    net.certainty = data["certainty"]
    net.certaintyRate = data["certaintyRate"]

    tBuffer = [[],[],[],[]]
    vBuffer = [[],[],[],[]]
    if not lazy:
        p = input_handling.readConfig()
        for i in range(4):
            tBuffer[i] = file_IO.decompressGames(file_IO.readGames('data/tBuffer' + str(i) + '.csv', p))
            vBuffer[i] = file_IO.decompressGames(file_IO.readGames('data/vBuffer' + str(i) + '.csv', p))
    
    return (net, tBuffer, vBuffer)
             
                
#   Both functions work elementwise on 2D numpy arrays
def leakyReLU(x):
    assert len(x.shape) == 2, x.shape
    y = np.zeros(x.shape)
    for j in range(x.shape[1]):
        for i in range(x.shape[0]):
            if x[i][j] >= 0:
                y[i][j] = x[i][j]
            else:
                y[i][j] = 0.01 * x[i][j]
    return y
            

def d_dx_leakyReLU(x):
    y = np.ones(x.shape)
    for j in range(x.shape[1]):
        for i in range(x.shape[0]):    
            if x[i][j] < 0:
                y[i][j] = 0.01
    return y
