import sys
import Network
import network_helper
import input_handling
import Traversal
import traverse_io
import Game
import misc

import random
import numpy as np
from scipy.special import expit

#   Given a network, asks the user for training hyper-parameters,
#   trains the network, and asks what to do next.
def trainOption(slowNet, tBuffer=[], vBuffer=[]): 
    p = input_handling.readConfig(0)
 
    #   traverseCount
    messDef = "Enter the number of sets of tree traversals to perform: "
    cond = 'var > 0'
    messOnErr = "Not a valid input."
    p['traverseCount'] = input_handling.getUserInput(messDef, messOnErr, 'int', cond)
        
    #   Set fastNet to slowNet
    fastNet = slowNet.copy()
        
    iterNum = 0
    numEps = p['traverseCount'] * p['updatePeriod']
    for i in range(numEps):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print('\tStarting episode ', i+1, ' of ', numEps, '!', sep='')
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            
        #   Add checkmates from file
        if i % p['updatePeriod'] == 0:  
            temp = network_helper.readGames('data/checkmates_t.csv')
            tGames = network_helper.decompressGames(temp)
            tBuffer += misc.divvy(tGames, p['fracFromFile'], False)[0]
            #print("Adding", len(tGames)*p['fracFromFile'], "games to tBuffer...")

            temp = network_helper.readGames('data/checkmates_v.csv')
            vGames = network_helper.decompressGames(temp)
            fracForV = len(tGames) * p['fracFromFile'] * p['fracValidation'] / ((1 - p['fracValidation']) * len(vGames))
            vBuffer += misc.divvy(vGames, fracForV, False)[0]
            #print("Adding", len(vGames)*fracForV, "games to vBuffer...")

        #   Randomly separate examples into training and validation buffers
        temp = misc.divvy(Traversal.full_broad(slowNet), p['fracValidation'])
        vBuffer += temp[0]
        tBuffer += temp[1]
        fastNet.experience += len(temp[1])

        #   QC stats for the examples generated
        mags = [abs(x[1] - 0.5) for x in tBuffer+vBuffer]
        avMag = sum(mags) / len(mags)
        avDev = sum([abs(x - avMag) for x in mags])/len(mags)
        print("--- Stats on examples generated ---")
        print("Number of t-examples:", len(tBuffer))
        print("Mean magnitude:", avMag)
        print("Average mag of deviation:", avDev)
        print("Mean reward:", sum([x[1] for x in tBuffer+vBuffer]) / len(mags))

        if (i + 1) % p['updatePeriod'] == 0:       
            #   Train on data in the buffer  
            fastNet.train(tBuffer, vBuffer, p)
            iterNum += 1

            print('------------------------------')
            print('Syncing slowNet to fastNet...')
            print('------------------------------')
            #   Adjust slowNet's expected input mean and variances for each layer.
            #   Then drop a fraction of the buffers
            fastNet.setPopStats(tBuffer + vBuffer)
            slowNet = fastNet.copy()
            if i < numEps-1:
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
    net.print()

messDef = 'Which of the following would you like to do:\n'
options = ['Train the current network including new data',
            'Show a sample game for the current network', 'Save the current network',
            'Check for dead neurons using current data', 'Train the network from previous data only',
            'Write existing novel checkmates to file', 'Run new games to generate checkmate positions']
for i, opt in enumerate(options):
    messDef += '(' + str(i+1) + ') ' + opt + '\n'
messOnErr = "Not a valid choice"

choice = 1
tBuffer, vBuffer = [], []
while choice > 0 and choice < len(options):
    net.print()
    choice = input_handling.getUserInput(messDef, messOnErr, 'int', 'True')

    if choice == 1:
        #   Keep a fraction of examples
        if len(tBuffer) > 0 and len(vBuffer) > 0:
            p = input_handling.readConfig(0)
            tBuffer = misc.divvy(tBuffer, 1 - p['memDecay'], False)[0]
            vBuffer = misc.divvy(vBuffer, 1 - p['memDecay'], False)[0]
    
        net, tBuffer, vBuffer = trainOption(net, tBuffer, vBuffer)
    elif choice == 2:
        print("Generating the current network's 'best' game...")
        net.showGame()
    elif choice == 3:
        filename = input("Name a file to save the network to: ")
        net.save('nets/' + filename)
        print("Saved. Continuing...")
    elif choice == 4:
        network_helper.dead_neurons(net, tBuffer+vBuffer)
    elif choice == 5:
        p = input_handling.readConfig(0)
        net.train(tBuffer, vBuffer, p)
    elif choice == 6:
        #   Find a way to not hardcode these?
        filepath = 'data/checkmates.csv'
        tol = 0.001
        p = input_handling.readConfig(1) # get mate reward
        temp = expit(p['mateReward']) - 0.5 - tol
        compressedGs = [network_helper.compressNNinput(g[0]) + [g[1]] for g in tBuffer+vBuffer if abs(g[1] - 0.5) > temp]
        novelGs = network_helper.filterByNovelty(compressedGs, filepath)
        network_helper.writeCheckmates(novelGs, filepath)
        print("Wrote", len(novelGs), "positions to file.")
    elif choice == 7:
        p = input_handling.readConfig(1) # get mate reward
        
        messDef = "Generate how many checkmate positions? "
        messOnErr = "Not a valid input."
        cond = 'var > 0'
        numPos = input_handling.getUserInput(messDef, messOnErr, 'int', cond)

        messDef = "Add to training (t) or validation (v) position file? "
        messOnErr = "Invalid input."
        fileChoice = input_handling.getUserInput(messDef, messOnErr, 'str', 'var == "t" or var == "v"')

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
            examples.append(network_helper.compressNNinput(temp[0]) + [expit(r)])
            examples.append(network_helper.compressNNinput(temp[1]) + [expit(-1 * r)])
        novelGames = network_helper.filterByNovelty(examples, filename)
        network_helper.writeCheckmates(novelGames, filename)
