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

class Network:
    def __init__(self, layers, weights=[], beta=[], gamma=[], popMean=[], popVar=[], tCosts=[], vCosts=[], age=0, experience=0, certainty=0):
        #   Weights and biases "beta"
        if len(weights) == 0:
            temp = [839] + layers
            self.weights = [np.random.randn(y, x)/np.sqrt(x)
                for x, y in zip(temp[:-1], temp[1:])]
        else:
            self.weights = weights
        if len(beta) == 0:
            self.beta = [np.full((x, 1), 0.1) for x in layers]
        else:
            self.beta = beta
            
        #   Ideal input variances "gamma"
        if len(gamma) == 0:
            self.gamma = [np.ones((x, 1)) for x in layers]
        else:
            self.gamma = gamma

        #   Population mean, variance, and stddev for inputs to each layer
        if len(popMean) == 0:
            self.popMean = [np.full((x, 1), -0.1) for x in layers]
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

        #   For momentum term in SGD
        self.last_dC_dw = [np.zeros(lay.shape) for lay in self.weights]
        self.last_dC_db = [np.zeros(lay.shape) for lay in self.beta]
        self.last_dC_dg = [np.zeros(lay.shape) for lay in self.gamma]

        #   For analysis of cost vs. epoch after training
        self.tCosts = tCosts
        self.vCosts = vCosts

        self.age = age  # number of training steps
        self.experience = experience # number of unique training examples seen
        self.certainty = certainty # see README- a measure of how much observed recent events match expected ones

    def copy(self):
        weights = [lay.copy() for lay in self.weights]
        beta = [lay.copy() for lay in self.beta]
        gamma = [lay.copy() for lay in self.gamma]
        popMean = [lay.copy() for lay in self.popMean]
        popVar = [lay.copy() for lay in self.popVar]
        
        return Network(self.layers, weights, beta, gamma, popMean, popVar, self.tCosts, self.vCosts, self.age, self.experience, self.certainty)
        
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
            #print(self.weights[lay].shape, a.shape, self.popMean[lay].shape, self.popDev[lay].shape)
            z = (self.weights[lay] @ a - self.popMean[lay]) * self.gamma[lay] / self.popDev[lay] + self.beta[lay]
            a = leakyReLU(z)
        #print(self.weights[-1].shape, a.shape, self.popMean[-1].shape, self.popDev[-1].shape)
        z = (self.weights[-1] @ a - self.popMean[-1]) * self.gamma[-1] / self.popDev[-1] + self.beta[-1]
        return expit(z)

    #   Inference for training on all members of a batch simultaneously
    def ff_track(self, aInput):
        assert len(aInput.shape) == 2 and aInput.shape[0] == 839, aInput.shape
        z, zNorm = [], []
        a = [aInput]
        for lay in range(len(self.layers) - 1):
            z.append(self.weights[lay] @ a[lay])
            zNorm.append(network_helper.normalize(z[lay], self.gamma[lay], self.beta[lay], self.eps))
            a.append(leakyReLU(zNorm[-1]))
        z.append(self.weights[-1] @ a[-1])
        zNorm.append(network_helper.normalize(z[-1], self.gamma[-1], self.beta[-1], self.eps))
        a.append(expit(zNorm[-1]))
        
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
        updatePer = p['updatePeriod']
        numCPUs = os.cpu_count()
        numBatches = int(len(games) / bs)
        
        #   Training via SGD
        if p['mode'] >= 2:
            start_time = time.time()
        pool = Pool()
        
        for epoch in range(epochs):
            random.shuffle(games)

            #   Baseline calculation of cost on training, validation data
            if epoch == 0:
                self.tCosts.append(self.totalCost(games))
                self.vCosts.append(self.totalCost(vGames))

            #   Divide data into batches, which each are divided into chunks-
            #   computation of the gradients will be parallelized by chunk
            chunkedGames = network_helper.toBatchChunks(games, bs, numCPUs)
            
            for i in range(numBatches):
                ###############################################################
                #   Submit the multicore 'job' and average returned gradients
                ###############################################################
                inList = [(self, chunk, p) for chunk in chunkedGames[i]]
                gradients_list = pool.map_async(train_thread, inList).get()

                #   Add up gradients by batch and parameter
                gradient = gradients_list[0]
                for j in range(1,len(gradients_list)):
                    #   Loop through each parameter (eg. weights, beta, etc)
                    for k in range(len(gradients_list[0])):
                        gradient[k] += gradients_list[j][k]

                ###############################################################
                #   Update parameters via SGD with momentum
                ###############################################################
                for lay in range(len(self.weights)):
                    self.last_dC_dw[lay] = gradient[0][lay] + p['mom'] * self.last_dC_dw[lay]
                    self.weights[lay] -= nu * self.last_dC_dw[lay]

                    self.last_dC_db[lay] = gradient[1][lay].reshape((-1,1)) + p['mom'] * self.last_dC_db[lay]
                    self.beta[lay] -= nu * self.last_dC_db[lay]

                    self.last_dC_dg[lay] = gradient[2][lay].reshape((-1,1)) + p['mom'] * self.last_dC_dg[lay]
                    self.gamma[lay] -= nu * self.last_dC_dg[lay]
                

            #   Compute ending cost on all training and validation examples
            self.tCosts.append(self.totalCost(games))
            self.vCosts.append(self.totalCost(vGames))
            print("Finished epoch ", epoch+1, ".", sep="")

        pool.close()
        if p['mode'] >= 2:
            speed = (time.time() - start_time) / (epochs * numBatches * bs)
            print("Averaged", speed, "seconds per training example")

        print('Updating pop stats...')
        self.setPopStats(games + vGames, p)
        self.age += numBatches
        print("Done training.")
        
        #   Write cost analytics to file
        self.costToCSV(epochs)

    #   Backprop is performed at once on the entire batch, where:
    #       -a, z, and zNorm are lists of 2d np arrays, with index of axis 1 in each array
    #        corresponding to the training example, and each list element referring
    #        to a NN layer
    #       -y is a 2d np array following this format idea
    def backprop(self, z, zNorm, a, y, p):
        nu = p['nu']
        mom = p['mom']
        
        #   This particular partial is summed up differently and requries initialization
        dC_dw = [np.zeros(self.weights[i].shape) for i in range(len(self.weights))]
        dC_dg = [np.zeros(self.gamma[i].shape) for i in range(len(self.gamma))]
        dC_db = [np.zeros(self.beta[i].shape) for i in range(len(self.beta))]

        batchSize = y.shape[1]
        assert batchSize > 0, "Attempted to do backprop on a batch of size 0"
        for i in range(len(z)):
            ###########################################################################
            #   First compute up to dC_dzNorm, the partial w/ respect to the batch
            #   normalized input, for the current layer
            ###########################################################################
            if i == 0:  # output layer
                dC_dzNorm = a[-1] - y   # assumes cross-entropy loss and sigmoid activation; 2D
            else:
                dC_dzNorm = d_dx_leakyReLU(zNorm[-1-i]) * np.dot(self.weights[-1*i].T, dC_dz)

            #print("dC_dzNorm.shape:", dC_dzNorm.shape)
            #   Sum up partials w/ respect to gamma and beta over the batch, and divide by batchSize
            dC_dg[-1-i] = np.sum(dC_dzNorm * (zNorm[-1-i] - self.beta[-1-i]) \
                                 / self.gamma[-1-i], axis = 1) / p['batchSize']
            #print("dC_dg[-1-1].shape:", dC_dg[-1-i].shape)
            dC_db[-1-i] = np.sum(dC_dzNorm, axis=1) / p['batchSize']
            #print("dC_db[-1-1].shape:", dC_db[-1-i].shape)
                
            ###########################################################################
            #   Compute dzNorm/dz (account for batch normalization in gradient flow)
            ###########################################################################
            bStats = network_helper.batchStats(z[-1-i])
            
            firstTerm = self.gamma[-1-i].flatten() /(batchSize * np.sqrt(bStats[0] + self.eps)) # 1D
            firstTerm = np.dot(firstTerm.reshape(-1,1), np.ones((1, batchSize)))    # 2D
            #print("firstTerm.shape:", firstTerm.shape)
            secTerm = batchSize - 1 - bStats[1] / np.dot(bStats[0].reshape(-1,1), np.ones((1, batchSize))) # 2D
            #print("secTerm.shape:", secTerm.shape)

            #   The term in parenthesis is dzNorm/dz (2D)
            dC_dz = dC_dzNorm * (firstTerm * secTerm)

            ###########################################################################
            #   Compute the final partials dC_dw, averaging over the batch
            ###########################################################################
            
            #   Sum up the partials w/ respect to the weights one training example at a time
            for j in range(batchSize):
                dC_dw[-1-i] = dC_dw[-1-i] + np.dot(dC_dz[:,j].reshape((-1,1)), a[-2-i][:,j].reshape((1,-1)))
            dC_dw[-1-i] = dC_dw[-1-i] / p['batchSize'] + p['weightDec'] * self.weights[-1-i]

        return [dC_dw, dC_db, dC_dg]

    #   Given the entire set of training examples, returns the average cost per example
    def totalCost(self, games):
        bGames = np.array([g[0].flatten() for g in games]).T
        bOuts = np.array([g[1] for g in games]).reshape(1,-1)
        
        a = self.ff_track(bGames)[2][-1]
        cost = -1 * np.sum(bOuts * np.log(a) + (1 - bOuts) * np.log(1 - a)) / len(games)

        return cost

    #   Return a numpy array of shape (len(data), ), corresponding to the loss for
    #   each data point/ training example in data.
    def individualCosts(self, data):
        inBatch = np.array([x[0].flatten() for x in data]).T
        labels = np.array([x[1] for x in data]).reshape(1,-1)

        outBatch = self.ff_track(inBatch)[2][-1]
        costs = -1 * labels * np.log(outBatch) + (1 - labels) * np.log(1 - outBatch)

        return costs.flatten()

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
    def setPopStats(self, pop, p, method=1):
        #   naive computation of population statistics: compute the mean and variance
        #   of the entire dataset
        if method == 0:
            pop = np.array([g[0].flatten() for g in pop]).T
            z = self.ff_track(pop)[0]
            popMean, popVar = [], []
            for lay in z:
                bStats = network_helper.batchStats(lay)
                popMean.append(bStats[2].reshape((-1,1)))
                popVar.append(bStats[0].reshape((-1,1)))

        #   "unbiased" estimate of population statistics: compute mean and variance
        #   in batches and average over these, as suggested in https://arxiv.org/pdf/1502.03167.pdf
        else:
            bs = p['batchSize']
            numCPUs = os.cpu_count()
            numBatches = int(len(pop) / bs)

            #   Compute stats in chunks (so pop stats are computed in a consistent way as stats
            #   are used during training)- this is equivalent to using a batch size of bs/numCPUs
            random.shuffle(pop)
            chunkedData = network_helper.toBatchChunks(pop, bs, numCPUs)

            pool = Pool()

            popMean = [np.zeros((lay, 1)) for lay in self.layers]
            popVar = [np.zeros((lay, 1)) for lay in self.layers]
            for i in range(numBatches):
                #   This is a list of tuples of lists: popStatList[chunkNum[statType[layerNum]]] is
                #   a numpy array representing the mean or variance of the activations for one layer
                #   from feeding forward one chunk
                inList = [(self, x[0]) for x in chunkedData[i]]
                popStatList = pool.map_async(network_helper.pop_stat_thread, inList).get()

                for chunk in popStatList:
                    for lay in range(len(self.layers)):
                        popMean[lay] += chunk[0][lay] / (numBatches * numCPUs)
                        popVar[lay] += chunk[1][lay] / (numBatches * numCPUs)

            pool.close()                       
                
        self.popMean = popMean
        self.popVar = popVar
        self.popDev = [np.sqrt(np.add(lay, self.eps)) for lay in popVar]

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

    def save(self, filename):
        data = {"layers": self.layers,
                "weights": [w.tolist() for w in self.weights],
                "beta": [b.tolist() for b in self.beta],
                "gamma": [g.tolist() for g in self.gamma],
                "popMean": [m.tolist() for m in self.popMean],
                "popVar": [v.tolist() for v in self.popVar],
                "age": self.age,
                "experience": self.experience,
                "certainty": self.certainty}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

def train_thread(inTuple):
    net = inTuple[0]
    batch = inTuple[1]
    p = inTuple[2]
    
    z, zNorm, a = net.ff_track(batch[0])
    return net.backprop(np.array(z), np.array(zNorm), np.array(a), batch[1], p)
      
def load(filename):
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
    
    return net
             
                
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
