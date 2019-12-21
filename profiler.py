import Network
import network_helper
import file_IO
import input_handling
import Traversal
import Game
import misc
import q_learn

import sys
import os
import shutil
import csv
import random
import numpy as np
from scipy.special import expit, logit
import cProfile

def do_train(arg_list):
    #   Given a network, asks the user for training hyper-parameters,
    #   trains the network, and asks what to do next.
    net, tBuffer, vBuffer = arg_list
    p = input_handling.readConfig(2)
    numEps = 1
        
    for i in range(numEps):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print('\tStarting episode ', i+1, ' of ', numEps, '!', sep='')
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("------------------")
        print("Generating data...")
        print("------------------")

        #   Randomly separate examples into training and validation buffers
        temp = misc.divvy(q_learn.async_q_learn(net), p['fracValidation'])
        vBuffer += temp[0]
        tBuffer += temp[1]
        numGenExamples = len(temp[1])
        net.experience += numGenExamples

        #   Add checkmates from file
        temp = file_IO.readGames('data/checkmates_t.csv', p)
        tGames = file_IO.decompressGames(temp)
        fracToUse = p['fracFromFile'] * numGenExamples / (len(tGames) * (1 - p['fracFromFile']))
        tBuffer += misc.divvy(tGames, fracToUse, False)[0]
        if p['mode'] >= 2:
            print("Adding", int(len(tGames)*p['fracFromFile']), "games to tBuffer...")

        temp = file_IO.readGames('data/checkmates_v.csv', p)
        vGames = file_IO.decompressGames(temp)
        fracToUse = p['fracValidation'] * fracToUse * len(tGames) / (len(vGames) * (1 - p['fracValidation']))
        vBuffer += misc.divvy(vGames, fracToUse, False)[0]
        if p['mode'] >= 2:
            print("Adding", int(len(vGames)*fracToUse), "games to vBuffer...\n")
        elif p['mode'] == 1:
            print()

        #   QC stats for the examples generated
        if p['mode'] >= 1:
            temp = [[float(logit(x[1])) for x in tBuffer+vBuffer]]
            if p['mode'] >= 2:
                print("Writing reward values to csv...")
                filename = "visualization/rewards.csv"
                with open(filename, 'w') as rFile:
                    writer = csv.writer(rFile)
                    writer.writerows(temp)
                print("Done.")
            rewards = np.array(temp)
            mags = abs(rewards)
            print("--- Stats on examples generated ---")
            print("Number of t-examples:", len(tBuffer))
            print("Mean reward:", np.round_(np.mean(rewards), 5))
            print("Std. deviation:", np.round_(np.std(rewards), 5))
            print("Mean magnitude:", np.round_(np.mean(mags), 5), "\n")
      
        #   Train on data in the buffer  
        net.train(tBuffer, vBuffer, p)

        #   Adjust net's expected input mean and variances for each layer.
        #   Then drop a fraction of the buffers
        if i < numEps-1:
            if p['mode'] >= 2:
                print("Filtering buffers to", round(1 - p['memDecay'], 4), "times their current size...")
            tBuffer = misc.divvy(tBuffer, 1 - p['memDecay'], False)[0]
            vBuffer = misc.divvy(vBuffer, 1 - p['memDecay'], False)[0]

    return (net, tBuffer, vBuffer)

net, tBuffer, vBuffer = Network.load('nets/8deep5')

net.print()
print('tBuffer and vBuffer sizes: ', len(tBuffer), ',', len(vBuffer))

cProfile.run('do_train([net,tBuffer,vBuffer])')

