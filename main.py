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

#   Given a network, asks the user for training hyper-parameters,
#   trains the network, and asks what to do next.
def trainOption(slowNet, tBuffer=[], vBuffer=[]): 
    p = input_handling.readConfig(2)
 
    #   traverseCount
    messDef = "Enter the number of sets of tree traversals to perform: "
    cond = 'var > 0'
    messOnErr = "Not a valid input."
    p['traverseCount'] = input_handling.getUserInput(messDef, messOnErr, 'int', cond)
        
    #   Set fastNet to slowNet
    fastNet = slowNet.copy()

    #   Determine if/ how many episodes to delay training in order to fill up the
    #   buffers to above the specified fraction (p['delayFrac']) of the limiting capacity
    if len(tBuffer) == 0:
        delayPeriod = max(0, int(np.ceil(p['delayFrac'] / p['memDecay']) - 1))
    else:
        delayPeriod = 0
    if p['mode'] >= 2:
        if len(tBuffer) == 0:
            if delayPeriod > 0:
                outStr = str(delayPeriod) + " episode(s) will be added to what was specified, in" +\
                         " order to get data buffers up to " + str(p['delayFrac']) + " times" +\
                         " their limiting size. To suppress this behavior, set delayFrac to 0" +\
                         " in config.txt."
            else:
                outStr = "No episodes needed to be added to what was specified, as the first episode" +\
                         " brings the buffers up to a sufficiently large size."
            print(outStr)
        else:
            print("No episodes will be added, as the data buffers are non-empty.")
        
    numEps = p['traverseCount'] * p['updatePeriod'] + delayPeriod
    for i in range(numEps):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print('\tStarting episode ', i+1, ' of ', numEps, '!', sep='')
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("------------------")
        print("Generating data...")
        print("------------------")

        #   Randomly separate examples into training and validation buffers
        #temp = misc.divvy(Traversal.full_broad(slowNet), p['fracValidation'])
        temp = misc.divvy(q_learn.aync_q_learn(slowNet), p['fracValidation'])
        vBuffer += temp[0]
        tBuffer += temp[1]
        numGenExamples = len(temp[1])
        fastNet.experience += numGenExamples

        #   Add checkmates from file
        if i % p['updatePeriod'] == 0: 
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

        if (i + 1) % p['updatePeriod'] == 0 and i  >= delayPeriod:       
            #   Train on data in the buffer  
            fastNet.train(tBuffer, vBuffer, p)

            print('Syncing slowNet to fastNet...')

            #   Adjust slowNet's expected input mean and variances for each layer.
            #   Then drop a fraction of the buffers
            fastNet.certainty = slowNet.certainty
            if i < numEps-1:
                slowNet = fastNet.copy()
                if p['mode'] >= 2:
                    print("Filtering buffers to", 1 - p['memDecay'], "times their current size...")
                tBuffer = misc.divvy(tBuffer, 1 - p['memDecay'], False)[0]
                vBuffer = misc.divvy(vBuffer, 1 - p['memDecay'], False)[0]

    return (fastNet, tBuffer, vBuffer)

def analyzeOption(network):
    return

#   Main -----------------------------------------------------------------------

messDef = "Load (l) network or create new (n)? "
messOnErr = "Not a valid option."
cond = "var == 'n' or var == 'l'"
choice = input_handling.getUserInput(messDef, messOnErr, 'str', cond)
if choice == "n":
    #   Define network architecture and initialize Network object
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

    net = Network.Network(layers)
elif choice == "l":
    filename = input("Load from what file? ")
    net = Network.load('nets/' + filename)
    print("Loaded successfully.")

messDef = 'Which of the following would you like to do:\n'
options = ['Train the current network including new data',
            'Show a sample game for the current network', 'Save the current network',
            'Check for dead neurons using current data', 'Train the network from previous data only',
            'Write existing novel checkmates to file', 'Run new games to generate checkmate positions',
            'Write the N least and greatest-loss positions to file']
for i, opt in enumerate(options):
    messDef += '(' + str(i+1) + ') ' + opt + '\n'
messDef += 'Enter 0 to exit: '
messOnErr = "Not a valid choice"

choice = 1
tBuffer, vBuffer = [], []
while choice > 0 and choice <= len(options):
    net.print()
    choice = input_handling.getUserInput(messDef, messOnErr, 'int', 'var >= 0 and var <= 8')
    print()

    if choice == 1:
        #   Keep a fraction of examples
        if len(tBuffer) > 0 and len(vBuffer) > 0:
            p = input_handling.readConfig(2)
            tBuffer = misc.divvy(tBuffer, 1 - p['memDecay'], False)[0]
            vBuffer = misc.divvy(vBuffer, 1 - p['memDecay'], False)[0]
    
        net, tBuffer, vBuffer = trainOption(net, tBuffer, vBuffer)
    elif choice == 2:
        p = input_handling.readConfig(2)
        print("Generating the current network's 'best' game...")
        net.showGame(verbose = p['mode'] >= 2)
    elif choice == 3:
        filename = input("Name a file to save the network to: ")
        net.save('nets/' + filename)
        print("Saved. Continuing...")
    elif choice == 4:
        network_helper.net_activity(net, tBuffer+vBuffer)
    elif choice == 5:
        p = input_handling.readConfig(2)
        net.train(tBuffer, vBuffer, p)
    elif choice == 6:
        p = input_handling.readConfig(0) # get mate reward
        
        messDef2 = "Add to training (t) or validation (v) position file? "
        messOnErr = "Invalid input."
        fileChoice = input_handling.getUserInput(messDef2, messOnErr, 'str', 'var == "t" or var == "v"')

        #   Find a way to not hardcode this?
        tol = 0.001
        temp = expit(p['mateReward']) - 0.5 - tol

        #   Select only training or only validation buffer: this prevents writing checkmates from checkmates_t.csv to
        #   checkmates_v.csv and vice versa.
        if fileChoice == 't':
            filename = 'data/checkmates_t.csv'
            compressedGs = [file_IO.compressNNinput(g[0]) + [g[1]] for g in tBuffer if abs(g[1] - 0.5) > temp]
        else:
            filename = 'data/checkmates_v.csv'
            compressedGs = [file_IO.compressNNinput(g[0]) + [g[1]] for g in vBuffer if abs(g[1] - 0.5) > temp]
               
        novelGs = file_IO.filterByNovelty(compressedGs, filename, p)
        file_IO.writeCheckmates(novelGs, filename)
        print("Wrote", len(novelGs), "positions to file.")
        
    elif choice == 7:
        p = input_handling.readConfig(0) # get mate reward
        
        messDef2 = "Generate how many checkmate positions? "
        messOnErr = "Not a valid input."
        cond = 'var > 0'
        numPos = input_handling.getUserInput(messDef2, messOnErr, 'int', cond)

        messDef2 = "Add to training (t) or validation (v) position file? "
        messOnErr = "Invalid input."
        fileChoice = input_handling.getUserInput(messDef2, messOnErr, 'str', 'var == "t" or var == "v"')

        if fileChoice == 't':
            filename = 'data/checkmates_t.csv'
        else:
            filename = 'data/checkmates_v.csv'

        examples = []
        for i in range(numPos):
            g = Game.Game()
            bestMoves = Traversal.initializeGame(net, True)[0]
            for m in bestMoves[:-1]:
                g.doMove(m)
            temp = g.toNN_vecs()
            g.doMove(bestMoves[-1]) # to produce the proper result
            r = g.gameResult * p['mateReward']
            assert abs(g.gameResult) == 1, g.gameResult
            examples.append(file_IO.compressNNinput(temp[0]) + [expit(r)])
            examples.append(file_IO.compressNNinput(temp[1]) + [expit(-1 * r)])
        novelGames = file_IO.filterByNovelty(examples, filename, p)
        file_IO.writeCheckmates(novelGames, filename)
    elif choice == 8:
        p = input_handling.readConfig()
        
        messDef2 = "Value of N? "
        messOnErr = "Large or negative values not supported."
        cond = 'var > 0 and var < 20'
        N = input_handling.getUserInput(messDef2, messOnErr, 'int', cond)

        print("Computing costs and writing positions...")

        allData = tBuffer + vBuffer
        costs = net.individualCosts(allData)

        #   Get rid of any old positions (to cover the case where the last choice
        #   of N is larger than the current choice)
        shutil.rmtree('visualization/edge_positions')
        os.makedirs('visualization/edge_positions')

        #   Write positions for top N largest costs
        for i, index in enumerate(misc.topN(costs, N)):
            filename = "visualization/edge_positions/highest_" + str(i+1) + ".fen"
            position = file_IO.compressNNinput(allData[index][0])
            file_IO.toFEN(position, filename, p['mode'] >= 2)

        #   Write positions for top N smallest costs
        for i, index in enumerate(misc.topN(-1 * costs, N)):
            filename = "visualization/edge_positions/lowest_" + str(i+1) + ".fen"
            position = file_IO.compressNNinput(allData[index][0])
            file_IO.toFEN(position, filename, p['mode'] >= 2)

        print("Done. See 'visualization/edge_positions/'.")
