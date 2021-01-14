import numpy as np
from scipy.special import expit, logit
import random
import csv
import os
from multiprocessing import Pool

import Network
import network_helper
import input_handling
import main

#   Variant of Network.train
def train(net, games, p, min_lr, max_lr):
    losses = []
    learn_rates = []

    p['nu'] = min_lr
    
    bs = p['batchSize']
    epochs = p['epochs']
    numCPUs = os.cpu_count()
    numBatches = int(len(games) / bs)

    growth_factor = (max_lr / min_lr) ** (1 / (numBatches * p['epochs']))

    pool = Pool()

    for epoch in range(epochs):
        #   Divide data into batches, which each are divided into chunks-
        #   computation of the gradients will be parallelized by chunk
        random.shuffle(games)
        chunkedGames = network_helper.toBatchChunks(games, bs, numCPUs)

        #   Baseline calculation of loss on training data
        if len(losses) == 0:
            losses.append(net.totalCost(games, p))
            learn_rates.append(p['nu'])

        for i in range(numBatches):
            ###############################################################
            #   Submit the multicore 'job' and average returned gradients
            ###############################################################
            inList = [(net, chunk, p) for chunk in chunkedGames[i]]
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

            net.last_dC_dbias[0] = gradient[3][0].reshape((-1,1)) + p['mom'] * net.last_dC_dbias[0]
            net.biases[0] -= p['nu'] * net.last_dC_dbias[0]
                            
            for lay in range(len(net.layers)):
                net.last_dC_dw[lay] = gradient[0][lay] + p['mom'] * net.last_dC_dw[lay]
                net.weights[lay] -= p['nu'] * net.last_dC_dw[lay]

                #   Batch norm-related parameters
                if lay < len(net.layers) - 1 and lay not in net.downSampLays:
                    net.last_dC_db[lay] = gradient[1][lay].reshape((-1,1)) + p['mom'] * net.last_dC_db[lay]
                    net.beta[lay] -= p['batchNormScale'] * p['nu'] * net.last_dC_db[lay]

                    net.last_dC_dg[lay] = gradient[2][lay].reshape((-1,1)) + p['mom'] * net.last_dC_dg[lay]
                    net.gamma[lay] -= p['batchNormScale'] * p['nu'] * net.last_dC_dg[lay]

            #   Compute loss and scale up learning rate
            losses.append(net.totalCost(games, p))
            learn_rates.append(p['nu'])
            p['nu'] *= growth_factor

    pool.close()

    return (losses, learn_rates)

def to_csv(losses, learn_rates):
    lossData = [["learnRate", "loss"]]

    for i in range(len(losses)):
        lossData.append([learn_rates[i], losses[i]])

    filename = "visualization/test_losses.csv"
    with open(filename, 'w') as lossFile:
        writer = csv.writer(lossFile)
        writer.writerows(lossData)

def train_thread(net, batch, p):  
    z, zNorm, a = net.ff_track(batch[0])
    
    return net.backprop(z, zNorm, a, batch[1], p)

################################################################
#   Main
################################################################

print('Loading training data...')
net, tBuffer, vBuffer = Network.load('nets/res2')

#print('Creating new network and reading config...')
#net = Network.Network(net.layers, net.blockWidth, net.blocksPerGroup)
data = main.collapseBuffer(tBuffer) + main.collapseBuffer(vBuffer)

print('Using', len(data), 'examples.')
p = input_handling.readConfig()
min_lr = 0.000001
max_lr = 10

print('Training at a range of learning rates...')
losses, learn_rates = train(net, data, p, min_lr, max_lr)
print('Writing results to csv...')
to_csv(losses, learn_rates)
print('Done.')
