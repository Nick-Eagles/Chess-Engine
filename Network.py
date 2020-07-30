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
        var_scale = 1.0
        alpha = 1.0

        self.activFun = reLU
        self.activFun_deriv = d_dx_reLU
        
        #   Layer numbers for those layers receiving activations from earlier layers, and those used in later layers, respectively
        self.resInputs = [i for i in range(len(layers)) if i > blockWidth - 1 and (i + int((i - 1) / (blockWidth * blocksPerGroup + 1))) % blockWidth == 0]
        self.resOutputs = [i - blockWidth for i in self.resInputs]
        self.downSampLays = [self.resOutputs[i] for i in range(len(self.resOutputs)) if i % blocksPerGroup == 0 and i > 0]
        
        #   Weights and biases "beta"
        if len(weights) == 0:
            temp = [839] + layers
            
            self.weights = []
            for i in range(len(layers)):
                #   Scale weights so that stddev of every layer's input is 1
                #if i - 1 in self.resInputs:
                    #scalar = np.sqrt(temp[i] * (alpha**2 + 1)) / var_scale
                #else:
                    #scalar = np.sqrt(temp[i]) / var_scale

                scalar = np.sqrt(2 / temp[i])
                    
                self.weights.append(np.random.randn(temp[i+1], temp[i]) * scalar)
            
        else:
            self.weights = weights

        if len(biases) == 0:
            self.biases = [np.zeros((layers[-1], 1))]
        else:
            self.biases = biases

        #   Ideal activation means "beta" for batch norm
        if len(beta) == 0:
            self.beta = []
            self.last_dC_db = []
            for i in range(len(layers) - 1):
                if i in self.downSampLays:
                    self.beta.append('Projection layers have no BN!')
                    self.last_dC_db.append('Projection layers have no BN!')
                else:
                    self.beta.append(np.zeros((layers[i], 1)))
                    self.last_dC_db.append(np.zeros((layers[i], 1)))
        else:
            self.beta = beta
            
        #   Ideal activation stddevs "gamma" for batch norm
        if len(gamma) == 0:
            self.gamma = []
            self.last_dC_dg = []
            for i in range(len(layers) - 1):
                if i in self.resInputs:
                    self.gamma.append(np.full((layers[i], 1), var_scale * alpha))
                    self.last_dC_dg.append(np.zeros((layers[i], 1)))
                elif i in self.downSampLays:
                    self.gamma.append('Projection layers have no BN!')
                    self.last_dC_dg.append('Projection layers have no BN!')
                else:
                    self.gamma.append(np.full((layers[i], 1), var_scale))
                    self.last_dC_dg.append(np.zeros((layers[i], 1)))

        else:
            self.gamma = gamma

        #   Population mean, variance, and stddev for inputs to each layer
        if len(popMean) == 0:
            self.popMean = []
            for i in range(len(layers)):
                if i in self.downSampLays:
                    self.popMean.append('Not computed for projection layers!')
                else:
                    self.popMean.append(np.zeros((layers[i], 1)))
        else:
            self.popMean = popMean
        if len(popVar) == 0:
            self.popVar = []
            for i in range(len(layers)):
                if i in self.downSampLays:
                    self.popVar.append('Not computed for projection layers!')
                else:
                    self.popVar.append(np.ones((layers[i], 1)))
        else:
            self.popVar = popVar
            
        self.eps = 0.00000001
        self.popDev = []
        for i in range(len(layers)):
            if i in self.downSampLays:
                self.popDev.append('Not computed for projection layers!')
            else:
                self.popDev.append(np.sqrt(np.add(self.popVar[i], self.eps)))
                

        #   List of lengths of hidden layers, in forward order
        self.layers = layers
        self.blockWidth = blockWidth
        self.blocksPerGroup = blocksPerGroup

        #   For momentum term in SGD
        self.last_dC_dw = [np.zeros(lay.shape) for lay in self.weights]
        self.last_dC_dbias = [np.zeros(self.biases[0].shape)]

        #   For analysis of cost vs. epoch after training
        self.tCosts = tCosts
        self.vCosts = vCosts

        self.age = age  # number of training steps
        self.experience = experience # number of unique training examples seen
        self.certainty = certainty # see README- a measure of how much observed recent events match expected ones
        self.certaintyRate = certaintyRate

    def copy(self):
        weights = [lay.copy() for lay in self.weights]
        biases = [self.biases[0].copy()]
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
    #   mean and variance. "a" is the input for a single example.
    def feedForward(self, a):
        assert len(a.shape) == 2 and a.shape[1] == 1, a.shape
        for lay in range(len(self.layers) - 1):
            if lay in self.downSampLays:
                #   This is a downsample layer, which simply performs a linear
                #   projection to match dimensionality
                a = self.weights[lay] @ a

            else:
                #   Batch norm
                a = (self.weights[lay] @ a - self.popMean[lay]) * self.gamma[lay] / self.popDev[lay] + self.beta[lay]

                #   Add identity for layers starting a residual block
                if lay in self.resInputs:
                    a += aLastBlock
                    
                #   Nonlinearity
                a = self.activFun(a)

            #   The start of a residual block has activations used [self.blockWidth] layers later
            if lay in self.resOutputs:
                aLastBlock = a.copy()

        #   Sigmoid without batch norm for the last layer
        a = expit(self.weights[-1] @ a + self.biases[0])
        assert a.shape == (self.layers[-1], 1), a.shape
        return a

    #   Inference for training on all members of a batch simultaneously
    def ff_track(self, aInput):
        assert len(aInput.shape) == 2 and aInput.shape[0] == 839, aInput.shape
        assert aInput.dtype == 'float64', aInput.dtype
        z, zNorm = [], []
        a = [aInput]
        for lay in range(len(self.layers)):
            #   Handle the 5 distinct types of layers separately
            if lay == len(self.layers) - 1:
                #   The last layer has biases, no batch norm, and sigmoid activation
                z.append(self.weights[lay] @ a[lay] + self.biases[0])
                zNorm.append('Last layer does not have BN!')
                a.append(expit(z[-1]))
            elif lay == 0:
                #   The first layer has batch norm (to account for arbitrary
                #   distributions of input data) but no nonlinearity
                z.append(self.weights[lay] @ a[lay])
                zNorm.append(network_helper.normalize(z[lay], self.gamma[lay], self.beta[lay], self.eps))
                a.append(zNorm[-1])
            elif lay in self.resInputs:
                #   Layers receiving shortcut connections have batch norm, the shortcut
                #   connection, then reLU of the resulting quantity. Note the last
                #   layer is not included in the set here
                z.append(self.weights[lay] @ a[lay])
                zNorm.append(network_helper.normalize(z[lay], self.gamma[lay], self.beta[lay], self.eps))
                a.append(self.activFun(zNorm[-1] + a[lay + 1 - self.blockWidth]))
            elif lay in self.downSampLays:
                #   Linear downsample layers
                temp = self.weights[lay] @ a[lay]
                z.append(temp)
                zNorm.append('Linear downsample layers do not have BN!')
                a.append(temp)
            else:
                #   "Typical" layers have batch norm and ReLU activation
                z.append(self.weights[lay] @ a[lay])
                zNorm.append(network_helper.normalize(z[lay], self.gamma[lay], self.beta[lay], self.eps))
                a.append(self.activFun(zNorm[lay]))
                         
        assert a[-1].shape == (self.layers[-1], aInput.shape[1]), a[-1].shape
        
        return (z, zNorm, a)

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
                    for lay in range(len(self.layers)):
                        print("----------------------------------")
                        print("  Layer ", lay, ":", sep="")
                        print("----------------------------------")
                        temp0 = round(nu * np.linalg.norm(gradient[0][lay]), 3)
                        temp1 = round(np.linalg.norm(self.weights[lay]), 2)
                        print("Magnitude of weight partial (weight value): ", temp0, " (", temp1, ")", sep="")
                        temp0 = round(nu * np.linalg.norm(gradient[1][lay]), 3)
                        temp1 = round(np.linalg.norm(self.beta[lay]), 2)
                        if lay < len(self.layers) - 1 and lay not in self.downSampLays:
                            print("Magnitude of beta partial (beta value): ", temp0, " (", temp1, ")", sep="")
                            temp0 = round(nu * np.linalg.norm(gradient[2][lay]), 3)
                            temp1 = round(np.linalg.norm(self.gamma[lay]), 2)
                            print("Magnitude of gamma partial (gamma value): ", temp0, " (", temp1, ")", sep="")

                self.last_dC_dbias[0] = gradient[3][0].reshape((-1,1)) + p['mom'] * self.last_dC_dbias[0]
                self.biases[0] -= nu * self.last_dC_dbias[0]
                            
                for lay in range(len(self.layers)):
                    self.last_dC_dw[lay] = gradient[0][lay] + p['mom'] * self.last_dC_dw[lay]
                    self.weights[lay] -= nu * self.last_dC_dw[lay]

                    #   Batch norm-related parameters
                    if lay < len(self.layers) - 1 and lay not in self.downSampLays:
                        self.last_dC_db[lay] = gradient[1][lay].reshape((-1,1)) + p['mom'] * self.last_dC_db[lay]
                        self.beta[lay] -= p['batchNormScale'] * nu * self.last_dC_db[lay]

                        self.last_dC_dg[lay] = gradient[2][lay].reshape((-1,1)) + p['mom'] * self.last_dC_dg[lay]
                        self.gamma[lay] -= p['batchNormScale'] * nu * self.last_dC_dg[lay]
                
            #   Approximate loss using batch statistics
            if p['mode'] >= 1 and epoch < epochs - 1:
                self.tCosts.append(self.totalCost(games, p))
                self.vCosts.append(self.totalCost(vGames, p))
            elif epoch == epochs - 1:
                self.setPopStats(games, p)
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
    #       -z, zNorm, and a are lists of 2d np arrays, with index of axis 1 in each array
    #        corresponding to the training example, and each list element referring
    #        to a NN layer
    #       -y is a 2d np array following this formatting style
    def backprop(self, z, zNorm, a, y, p):
        #   Sanity checks on number of layers
        assert len(z) == len(self.layers)
        assert len(zNorm) == len(self.layers)
        assert len(a) == len(self.layers) + 1

        assert a[0].shape == (839, y.shape[1]), a[0].shape
        assert a[-1].shape == y.shape, a[-1].shape

        if p['mode'] >= 2:
            bs = y.shape[1]
            assert a[0].shape == (839, bs), a[0].shape
            assert a[-1].shape == y.shape, a[-1].shape
            for i in range(len(self.layers)):
                if i not in self.resOutputs:
                    assert z[i].shape == (self.layers[i], bs), z[i].shape
                    if i < len(self.layers) - 1:
                        assert zNorm[i].shape == (self.layers[i], bs), zNorm[i].shape
                assert a[i+1].shape == (self.layers[i], bs), a[i+1].shape
        
        nu = p['nu']
        mom = p['mom']
        
        #   Initialize partials
        dC_dw = [np.zeros(self.weights[i].shape) for i in range(len(self.weights))]
        dC_dg, dC_db = [], []
        for i in range(len(self.gamma)):
            if i in self.downSampLays:
                dC_dg.append('Partials undefined')
                dC_db.append('Partials undefined')
            else:
                dC_dg.append(np.zeros(self.gamma[i].shape))
                dC_db.append(np.zeros(self.beta[i].shape))
                
        dC_dz = [np.zeros(z[i].shape) for i in range(len(z))]
        dC_dbias = [np.zeros(self.biases[0].shape)]

        batchSize = y.shape[1]
        assert batchSize > 0, "Attempted to do backprop on a batch of size 0"
        for i in range(len(z)):
            ###########################################################################
            #   First compute up to dC_dzNorm, the partial w/ respect to the batch
            #   normalized input, for the current layer
            ###########################################################################
            if i == 0:  # output layer
                dC_dz[-1] = a[-1] - y   # assumes cross-entropy loss and sigmoid activation; 2D
                dC_dbias[0] = np.sum(dC_dz[-1], axis=1) / p['batchSize']
            else:
                #################################################################
                #   Compute up to dC_da
                #################################################################
                if len(z) - 1 - i in self.resOutputs:
                    dC_da = np.dot(self.weights[-1*i].T, dC_dz[-1*i]) + dC_daRes * self.activFun_deriv(zNorm[-1-i+self.blockWidth] + a[-1-i])
                else:
                    dC_da = np.dot(self.weights[-1*i].T, dC_dz[-1*i])

                #################################################################
                #   Compute up to dC_dzNorm for layers with batch norm, and up
                #   to dC_dz for those without
                #################################################################
                if len(z) - 1 - i in self.resInputs:
                    #   Save the partial wrt the activation at layers receiving a shortcut connection
                    dC_daRes = dC_da.copy()

                    dC_dzNorm = dC_da * self.activFun_deriv(zNorm[-1-i] + a[-1-i-self.blockWidth])
                elif len(z) - 1 - i in self.downSampLays:
                    dC_dz[-1-i] = dC_da
                elif len(z) - 1 - i == 0:
                    dC_dzNorm = dC_da
                else:
                    dC_dzNorm = dC_da * self.activFun_deriv(zNorm[-1-i])

            ###########################################################################
            #   Account for batch norm, for applicable layers
            ###########################################################################

            #   Batch norm is not done for output layer or linear projection layers
            if i > 0 and len(z) - 1 - i not in self.downSampLays:
                #   Sum up partials w/ respect to gamma and beta over the batch, and divide by batchSize
                dC_dg[-1*i] = np.sum(dC_dzNorm * (zNorm[-1-i] - self.beta[-1*i]) \
                                     / self.gamma[-1*i], axis = 1) / p['batchSize']
                dC_db[-1*i] = np.sum(dC_dzNorm, axis=1) / p['batchSize']
                    
                ###########################################################################
                #   Compute dzNorm/dz (account for batch normalization in gradient flow)
                ###########################################################################

                ################
                # Orig method
                ################
                #bStats = network_helper.batchStats(z[-1-i])

                #firstTerm = self.gamma[-1*i].flatten() /(batchSize * np.sqrt(bStats[0] + self.eps)) # 1D
                #firstTerm = np.dot(firstTerm.reshape(-1,1), np.ones((1, batchSize)))    # 2D
                #secTerm = batchSize - 1 - bStats[1] / np.dot(np.add(bStats[0].reshape(-1,1), self.eps), np.ones((1, batchSize))) # 2D

                #   The term in parenthesis is dzNorm/dz (2D)
                #dC_dz[-1-i] = dC_dzNorm * (firstTerm * secTerm)

                ################
                # New method
                ################
                bStats = network_helper.batchStats(z[-1-i], self.gamma[-1*i], self.eps)

                partialSum = np.sum(dC_dzNorm * (-1 - bStats[0]), axis=1).reshape((-1,1))
                dC_dz[-1-i] = bStats[1] * (np.dot(partialSum, np.ones((1, batchSize))) + dC_dzNorm * batchSize)

            ###########################################################################
            #   Compute the final partials dC_dw, averaging over the batch
            ###########################################################################
            
            #   Sum up the partials w/ respect to the weights one training example at a time
            dC_dw[-1-i] = np.dot(dC_dz[-1-i], a[-2-i].T) / p['batchSize'] + (p['weightDec'] / os.cpu_count()) * self.weights[-1-i]

        return [dC_dw, dC_db, dC_dg, dC_dbias]

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
        labels = np.array([x[1].flatten() for x in data]).T

        outBatch = self.ff_track(inBatch)[2][-1]
        costs = -1 * (labels * np.log(outBatch) + (1 - labels) * np.log(1 - outBatch))
        #temp = np.mean((outBatch - labels) * (outBatch - labels), axis=1)
        #return float(np.sqrt((temp.T @ temp) / labels.shape[0]))

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
                    if lay not in self.downSampLays:
                        popMean[lay] += chunk[0][lay] / (numBatches * numCPUs)
                        popVar[lay] += chunk[1][lay] / (numBatches * numCPUs)

        pool.close()

        if p['mode'] >= 2:
            concDirMean = 0
            diffMagMean = 0
            concDirVar = 0
            diffMagVar = 0
        for i in range(len(self.popMean)):
            if i not in self.downSampLays:
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
            #   Note these are stats for the last layer where batch norm occurs
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
            print('  ', np.round_(lay[0][:5], 3))
        print('First 5 biases:')
        print('  ', np.round_(self.biases[0][:5], 3).T)
        print('First 5 beta:')
        for lay in range(len(self.beta)):
            if lay in self.downSampLays:
                print('  (Undefined)')
            else:
                print('  ', np.round_(self.beta[lay][:5], 3).T)
        print('First 5 gamma:')
        for lay in range(len(self.gamma)):
            if lay in self.downSampLays:
                print('  (Undefined)')
            else:
                print('  ', np.round_(self.gamma[lay][:5], 3).T)
        print('First 5 input means:')
        for lay in range(len(self.popMean)):
            if lay in self.downSampLays:
                print('  (Not computed)')
            else:
                print('  ', np.round_(self.popMean[lay][:5], 3).T)
        print('First 5 input vars:')
        for lay in range(len(self.popVar)):
            if lay in self.downSampLays:
                print('  (Not computed)')
            else:
                print('  ', np.round_(self.popVar[lay][:5], 3).T)
        print('Number of training steps total:', self.age)
        print('Unique examples seen: ~', self.experience, sep="")
        print('Moving certainty:', round(self.certainty, 3))
        print('Residual "input" layers:', self.resInputs)
        print('Residual "output" layers:', self.resOutputs)
        print('Linear downsample layers:', self.downSampLays)

    def save(self, tBuffer, vBuffer, filename):       
        beta, gamma, popMean, popVar = [], [], [], []
        for i in range(len(self.layers)):
            if i in self.downSampLays:
                beta.append([])
                gamma.append([])
                popMean.append([])
                popVar.append([])
            else:
                if i < len(self.layers) - 1:
                    beta.append(self.beta[i].tolist())
                    gamma.append(self.gamma[i].tolist())
                popMean.append(self.popMean[i].tolist())
                popVar.append(self.popVar[i].tolist())
                
        data = {"layers": self.layers,
                "weights": [w.tolist() for w in self.weights],
                "biases": [self.biases[0].tolist()],
                "beta": beta,
                "gamma": gamma,
                "popMean": popMean,
                "popVar": popVar,
                "age": self.age,
                "experience": self.experience,
                "certainty": self.certainty,
                "certaintyRate": self.certaintyRate,
                "blockWidth": self.blockWidth,
                "blocksPerGroup": self.blocksPerGroup,
                "tCosts": self.tCosts,
                "vCosts": self.vCosts}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

        #   Write each sub-buffer to separate file
        for i in range(4):
            file_IO.writeGames(tBuffer[i], 'data/tBuffer' + str(i) + '.csv', True, False)
            file_IO.writeGames(vBuffer[i], 'data/vBuffer' + str(i) + '.csv', True, False)

def train_thread(net, batch, p):  
    z, zNorm, a = net.ff_track(batch[0])
    
    return net.backprop(z, zNorm, a, batch[1], p)
      
def load(filename, lazy=False):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    
    net = Network(data["layers"], data["blockWidth"], data["blocksPerGroup"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(data["biases"]).reshape((-1,1))]

    net.beta, net.gamma, net.popMean, net.popVar, net.popDev = [], [], [], [], []
    for i in range(len(net.layers)):
        if i in net.downSampLays:
            net.beta.append(data["beta"][i])
            net.gamma.append(data["gamma"][i])
            net.popMean.append(data["popMean"][i])
            net.popVar.append(data["popVar"][i])
            net.popDev.append('Not computed for projection layers!')
        else:
            if i < len(net.layers) - 1:
                net.beta.append(np.array(data["beta"][i]).reshape((-1,1)))
                net.gamma.append(np.array(data["gamma"][i]).reshape((-1,1)))
            net.popMean.append(np.array(data["popMean"][i]).reshape((-1,1)))
            net.popVar.append(np.array(data["popVar"][i]).reshape((-1,1)))
            net.popDev.append(np.sqrt(np.add(net.popVar[-1], net.eps)).reshape((-1,1)))

    net.age = data["age"]
    net.experience = data["experience"]
    net.certainty = data["certainty"]
    net.certaintyRate = data["certaintyRate"]
    net.tCosts = data["tCosts"]
    net.vCosts = data["vCosts"]

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

def reLU(x):
    return np.maximum(0, x)

def d_dx_reLU(x):
    y = np.zeros(x.shape)
    y[x > 0] = 1

    return y
