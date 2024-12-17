import network_helper
import file_IO
import input_handling
import Traversal
import Game
import misc
import q_learn
import demonstration
import Session
import buffer
import policy
import policy_net

import sys
import os
import shutil
import csv
import numpy as np
from scipy.special import expit, logit
import time
from pyhere import here
from tensorflow.keras.models import load_model

sys.path.append(str(here('external')))
sys.path.append(str(here('experimental')))
import read_pgn


#   Given a network, asks the user for training hyper-parameters, trains the
#   network, and asks what to do next.
def trainOption(session, numEps=0): 
    p = input_handling.readConfig()

    net = session.net
    tBuffer = session.tBuffer
    vBuffer = session.vBuffer
 
    #   traverseCount
    if numEps == 0:
        messDef = "Enter the number of sets of tree traversals to perform: "
        cond = 'var > 0'
        messOnErr = "Not a valid input."
        numEps = input_handling.getUserInput(messDef, messOnErr, 'int', cond)
        
    for i in range(numEps):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print('\tStarting episode ', i+1, ' of ', numEps, '!', sep='')
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("------------------")
        print("Generating data...")
        print("------------------")

        #   Randomly separate examples into training and validation buffers
        temp = [misc.divvy(buff, p['fracValidation'])
                for buff in q_learn.async_q_learn(net)]
        for j in range(3):
            vBuffer[j] += temp[j][0]
            tBuffer[j] += temp[j][1]
        numGenExamples = len(temp[0][1])

        #   Add checkmates from file
        tGames = file_IO.readBuffer('data/tCheckmates.pkl.gz', p)[0]
        fracToUse = p['fracFromFile'] * numGenExamples / \
                    (len(tGames) * (1 - p['fracFromFile']))
        tBuffer[3] += misc.divvy(tGames, fracToUse, False)[0]
        if p['mode'] >= 2:
            print("Adding", int(len(tGames)*p['fracFromFile']),
                  "games to tBuffer...")

        vGames = file_IO.readBuffer('data/vCheckmates.pkl.gz', p)[0]
        fracToUse = p['fracValidation'] * fracToUse * len(tGames) / (len(vGames) * (1 - p['fracValidation']))
        vBuffer[3] += misc.divvy(vGames, fracToUse, False)[0]
        if p['mode'] >= 2:
            print("Adding", int(len(vGames)*fracToUse), "games to vBuffer...\n")
        elif p['mode'] == 1:
            print()

        #   QC stats for the examples generated
        if p['mode'] >= 1:
            temp = [[float(logit(x[1])) for x in tBuffer[0]+tBuffer[3]]]
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
            print("Number of t-examples:", sum([len(x) for x in tBuffer]))
            print("Mean reward:", np.round_(np.mean(rewards), 5))
            print("Std. deviation:", np.round_(np.std(rewards), 5))
            print("Mean magnitude:", np.round_(np.mean(mags), 5), "\n")
      
        #   Train on data in the buffer  
        network_helper.train(
            net, buffer.collapse(tBuffer), buffer.collapse(vBuffer), p
        )

        #   Drop a fraction of the buffers
        if i < numEps-1:
            if p['mode'] >= 2:
                print("Filtering buffers...")

            buffer.filter(tBuffer, vBuffer, p)


def analyzeOption(network):
    return
  
    
#   Main -----------------------------------------------------------------------

if __name__ == '__main__':
    messDef = "Load (l) network or create new (n)? "
    messOnErr = "Not a valid option."
    cond = "var == 'n' or var == 'l'"
    choice = input_handling.getUserInput(messDef, messOnErr, 'str', cond)
    if choice == "n":
        p = input_handling.readConfig()
        
        tBuffer = [[],[],[],[]]
        vBuffer = [[],[],[],[]]
        
        ##################################################################
        #   Define network architecture and initialize Network object
        ##################################################################

        #   Policy and value network or just value network?
        messDef = "Include policy in output (y/n)? "
        messOnErr = '"y" or "n" accepted only.'
        cond = 'var == "y" or var == "n"'
        temp = input_handling.getUserInput(messDef, messOnErr, 'str', cond)
        if temp == 'y':
            output_type = 'policy_value'
        else:
            output_type = 'value'
        
        net = policy_net.InitializeNet(p, output_type)
        session = Session.Session(tBuffer, vBuffer, net)
        
    elif choice == "l":
        filename = input("Load from what file? ")
        net = load_model(here('nets', filename))
        net.value_certainty = 1 # temporary workaround!

    messDef = 'Which of the following would you like to do:\n'
    options = [
        'Train the current network including new data',
        'Show a sample game for the current network',
        'Save the current network',
        'Check neuron activations using current data',
        'Train the network from previous data only',
        'Write existing novel checkmates to file',
        'Run new games to generate checkmate positions',
        'Write the N least and greatest-loss positions to file',
        'Play a game against the network',
        'Load an internal dataset',
        'Load an external datset'
    ]
    for i, opt in enumerate(options):
        messDef += '(' + str(i+1) + ') ' + opt + '\n'
    messDef += 'Enter 0 to exit: '
    messOnErr = "Not a valid choice"

    choice = 1
    while choice > 0 and choice <= len(options):
        net.summary()
        
        cond = 'var >= 0 and var <= ' + str(len(options))
        choice = input_handling.getUserInput(messDef, messOnErr, 'int', cond)
                                             
        print()

        if choice == 1:
            #   Keep a fraction of examples
            p = input_handling.readConfig()
            if len(session.tBuffer[0]) > 0 and len(session.vBuffer[0]) > 0:
                buffer.filter(session.tBuffer, session.vBuffer, p)
        
            trainOption(session)
        elif choice == 2:
            p = input_handling.readConfig()
            print("Generating the current network's 'best' game...")
            network_helper.bestGame(net, policy.getBestMoveTreeEG)
        elif choice == 3:
            dirname = 'nets/' + input("Name a file to save the network to: ")
            session.Save(dirname)
            print("Saved. Continuing...")
        elif choice == 4:
            network_helper.net_activity(
                net,
                buffer.collapse(session.tBuffer) + \
                    buffer.collapse(session.vBuffer)
            )
        elif choice == 5:
            p = input_handling.readConfig()
            network_helper.train(
                session.net,
                buffer.collapse(session.tBuffer),
                buffer.collapse(session.vBuffer),
                p
            )
        elif choice == 6:
            p = input_handling.readConfig()
            
            messDef2 = "Add to training (t) or validation (v) position file? "
            messOnErr = "Invalid input."
            fileChoice = input_handling.getUserInput(
                messDef2, messOnErr, 'str', 'var == "t" or var == "v"'
            )

            #   Find a way to not hardcode this?
            tol = 0.001
            temp = expit(p['mateReward']) - 0.5 - tol

            #   Select only training or only validation buffer: this prevents
            #   writing checkmates from checkmates_t.csv to checkmates_v.csv
            #   and vice versa.
            if fileChoice == 't':
                filename = 'data/checkmates_t.csv'
                compressedGs = [file_IO.compressNNinput(g[0]) + [g[1]]
                                for g in tBuffer if abs(g[1] - 0.5) > temp]
            else:
                filename = 'data/checkmates_v.csv'
                compressedGs = [file_IO.compressNNinput(g[0]) + [g[1]]
                                for g in vBuffer if abs(g[1] - 0.5) > temp]
                   
            novelGs = file_IO.filterByNovelty(compressedGs, filename, p)
            file_IO.writeGames(novelGs, filename)
            print("Wrote", len(novelGs), "positions to file.")
            
        elif choice == 7:
            p = input_handling.readConfig() # get mate reward
            
            messDef2 = "Generate how many checkmate positions? "
            messOnErr = "Not a valid input."
            cond = 'var > 0'
            numPos = input_handling.getUserInput(
                messDef2, messOnErr, 'int', cond
            )

            messDef2 = "Add to training (t) or validation (v) position file? "
            messOnErr = "Invalid input."
            fileChoice = input_handling.getUserInput(
                messDef2, messOnErr, 'str', 'var == "t" or var == "v"'
            )

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
                examples.append(
                    file_IO.compressNNinput(temp[0]) + [expit(r)]
                )
                examples.append(
                    file_IO.compressNNinput(temp[1]) + [expit(-1 * r)]
                )
            novelGames = file_IO.filterByNovelty(examples, filename, p)
            file_IO.writeGames(novelGames, filename)
        elif choice == 8:
            p = input_handling.readConfig()
            
            messDef2 = "Value of N? "
            messOnErr = "Large or negative values not supported."
            cond = 'var > 0 and var < 20'
            N = input_handling.getUserInput(messDef2, messOnErr, 'int', cond)

            print("Computing costs and writing positions...")

            allData = buffer.collapse(session.tBuffer) + \
                      buffer.collapse(session.vBuffer)
            costs = np.array([net.individualLoss([x]) for x in allData])

            #   Get rid of any old positions (to cover the case where the last
            #   choice of N is larger than the current choice)
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
        elif choice == 9:
            demonstration.interact(net)
        elif choice == 10:
            p = input_handling.readConfig()

            #   Get name of dataset
            messDef2 = "Name of dataset? "
            messOnErr = "Error."
            prefix = "data/" + input_handling.getUserInput(
                messDef2, messOnErr, 'str', 'True'
            )

            #   Ask whether to replace existing data
            messDef2 = "Replace existing data (y/n)? "
            messOnErr = "Only 'y' or 'n' accepted."
            replace = input_handling.getUserInput(
                messDef2, messOnErr, 'str', 'var == "y" or var == "n"'
            )
            replace = replace == 'y'
            
            temp_tBuffer = file_IO.readBuffer(prefix + '/tBuffer.pkl.gz', p)
            temp_vBuffer = file_IO.readBuffer(prefix + '/vBuffer.pkl.gz', p)

            if replace:
                session.tBuffer = temp_tBuffer
                session.vBuffer = temp_vBuffer
            else:
                for i in range(4):
                    session.tBuffer[i] += temp_tBuffer[i]
                    session.vBuffer[i] += temp_vBuffer[i]
        elif choice == 11:
            p = input_handling.readConfig()

            messDef2 = "Filename for the training data? "
            messOnErr = "Not a valid filename."
            t_filename = 'external/' + input_handling.getUserInput(
                messDef2, messOnErr, 'str', 'True'
            )

            messDef2 = "Loop through how many games total? "
            messOnErr = "Not a valid amount."
            num_games = input_handling.getUserInput(
                messDef2, messOnErr, 'int', 'var > 0'
            )

            messDef2 = "Number of games per chunk? "
            messOnErr = "Not a valid amount."
            chunk_size = input_handling.getUserInput(
                messDef2, messOnErr, 'int', 'var > 0'
            )

            num_chunks = int(num_games / chunk_size)

            #   Load validation games
            print('Loading validation data...')
            filename = 'external/2019_games_processed_v.txt'
            line_nums = list(range(201))
            session.vBuffer = read_pgn.load_games(
                filename, p, line_nums, session.net, certainty=False
            )
            print('Loaded', len(session.vBuffer[0]), 'positions.')

            for i in range(num_chunks):
                print("Processing chunk ", i+1, " of ", num_chunks, "!", sep='')
                start_time = time.time()

                line_nums = list(range(i * chunk_size, (i+1) * chunk_size))
                

                #   Read chunk into memory
                session.tBuffer = read_pgn.load_games(
                    t_filename, p, line_nums, session.net
                )

                elapsed = time.time() - start_time
                rate = round(len(session.tBuffer[0]) / elapsed, 2)
                
                print(
                    "Generated ", len(session.tBuffer[0]),
                    " training examples in ", round(elapsed, 2), " seconds (",
                    rate, "/sec)", sep=""
                )

                network_helper.train(
                    session.net,
                    buffer.collapse(session.tBuffer),
                    buffer.collapse(session.vBuffer),
                    p
                )
