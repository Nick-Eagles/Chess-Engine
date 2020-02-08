import board_helper
import input_handling
import Game

import os
import csv
import numpy as np
import copy

def writeTree(game, p):
    tDepth = p['tDepth']
    rDepth = p['rDepth']
    alpha = p['alpha']
    
    print("-----------------------------------")
    print("Beginning full breadth traversal...")
    print("-----------------------------------")

    m = board_helper.getLegalMoves(game)
    depth0len = len(m) # to save original number of moves from the base node

    ######################################################
    ###     Tree traversal
    ######################################################
    stack = [[m, game, 0]]
    if game.whiteToMove:
        writeStack = ['w']
    else:
        writeStack = ['b']
        
    nodeHops = 0
    while len(stack) > 0:
        numMoves = len(stack[-1][0])
        #   At the base node, record how many moves are left to monitor progress
        if len(stack) == 1:
            progress = int(100 * ((depth0len - numMoves) / depth0len))
            if progress > 0 and progress < 100:
                print("Explored an estimated ", progress, "% of the tree.", sep="")

        #   If there are still moves to explore from this node and not too deep
        if numMoves > 0 and len(stack) <= tDepth + rDepth:
            #   Copy the game, do the move, and observe the reward
            g = copy.deepcopy(stack[-1][1])
            reward = np.log(g.wValue / g.bValue)
            g.doMove(stack[-1][0].pop())
            reward = float(np.log(g.wValue / g.bValue) - reward)

            if g.gameResult == 17:
                #   Game continues, push another node to stack
                m = board_helper.getLegalMoves(g)
                stack.append([m, g, reward])
                nodeHops += 1
            else:
                reward = 5 * g.gameResult
                temp = g.toNN_inputs()
                writeStack.append((temp[0], reward))
                #writeStack.append((temp[1], reward * -1))
        else:
            #   Mark a node hop downward with a '#' string
            nodeHops += 1
            writeStack.append("#")
            layer = stack.pop()

            NN_inputs = layer[1].toNN_vecs()
            writeStack.append((NN_inputs[0], layer[2]))
            #writeStack.append((NN_inputs[1], -1 * layer[2]))

    ######################################################
    ###     Convert stack to writeable string, and write
    ######################################################
    finalStack = []
    for i in writeStack:
        if i == "#" or i == 'w' or i == 'b':
            finalStack.append(i)
        else:
            finalStack.append(i[0].flatten().tolist() + [i[1]])

    #   User input, determine whether to append or write new file
    messDef = "Enter a file to write the positions to (please add '.csv'): "
    cond = "len(var) >= 4 and var[-4:] == '.csv'"
    messOnErr = "Invalid file name."
    filename = input_handling.getUserInput(messDef, messOnErr, 'str', cond)
    filename = "data/" + filename
    filepath = os.getcwd() + "/" + filename
    try:
        gameFile = ''
        if os.path.exists(filepath):
            print("This file exists already; data will be appended.")
            gameFile = open(filename, 'a')
        else:
            print("This file does not exist; it will be created.")
            gameFile = open(filename, 'w')

        #   The actual writing of data
        with gameFile:
            writer = csv.writer(gameFile)
            writer.writerows(finalStack)
        print("Writing complete.")
    except:
        print("Encountered an error while opening or handling file")
    finally:
        gameFile.close()

def read_traversal_data(net, alpha, tDepth, rDepth, filename):
    #   Open and read file
    lines = []
    try:
        gameFile = open(filename, 'r')
        with gameFile:
            reader = csv.reader(gameFile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)

            for line in reader:
                lines.append(line)
        print("Read the file without errors.")
    except:
        print("Problem opening or reading file.")
    finally:
        gameFile.close()

    startColor = lines[0] == 'w'
    #   Traverse in the same way done in traverse.py
    tData = []
    depth = 0
    maxDepth = 0
    expRewards = [list() for i in range(rDepth + tDepth)]
    actRewards = [list() for i in range(rDepth + tDepth)]
    for line in lines[1:]:
        if depth > maxDepth:
            maxDepth = depth
        if line[0] != '#':
            #   Then store the reward data at the current depth's list
            actRew = line.pop()
            #print(actRew)
            actRewards[depth].append(actRew)
            lineAsArr = np.array([float(i) for i in line])
            expRew = net.feedForward(np.array(lineAsArr))
            expRewards[depth].append(expRew)
            depth += 1
        else:
            #   This signifies being done exploring all moves from a given node, since
            #   exploration is a hop of 1, and moving on requires jumping 2 hops down
            #   (or 1 before the end of this code segment). In this case pass weighted
            #   combination of reward down
            if maxDepth - depth == 1:
                #   Pass reward down
                r = toWeightedRew(net, expRewards[maxDepth], actRewards[maxDepth], alpha, depth%2 != startColor)
                actRewards[depth][-1] += r
                
                #   Clear the list of rewards we just added together
                expRewards.pop()
                actRewards.pop()
                maxDepth -= 1
                            
                #   Use the node we hopped down from as training data if it's not too deep
                if depth <= tDepth:
                    tData.append((expRewards[depth][-1], r))
                    
            #   Node hop down
            depth -= 1

    assert depth == 0, "Didn't return to 0 depth after traversal"
    return tData

def initializeBase(net):
    #   User input of remaining parameters   
    messDef = "Use config file for tree traversal settings (1 for true, 0 for false)? "
    cond = 'var == 0 or var == 1'
    messOnErr = "Not a valid choice."
    p  = {'useConfig': input_handling.getUserInput(messDef, messOnErr, 'int', cond)}

    if p['useConfig']:
        #   Literally assign variables as specified by the config
        assignments = input_handling.readConfig(1)
        for i in assignments:
            var = i[:i.index(" ")]
            val = i[i.index(" ")+1:]
            if "." in val:
                p[var] = float(val)
            else:
                p[var] = int(val)
    else:
        p = input_handling.getP(1)  

    #   Start and finish game normally
    print("-------------------------")
    print("Initializing base game...")
    print("-------------------------")
    game = Game.Game()
    bestInputs = list()
    bestMoves = list()
    captures = list()
    while (game.gameResult == 17):
        #   Find and do the best legal move
        legalMoves = board_helper.getLegalMoves(game)
        captCount = 0
        for m in range(len(legalMoves)):
            gameTuple = game.testMove(legalMoves[m])
            netInputs = board_helper.gameTupleToInput(gameTuple)
            val = net.feedForward(netInputs[0])    #   Shouldn't be a need to check inverted input also
                            
            #   weighted average of actual value and a random num in [-1,1]
            epsVal = (1 - p['epsilon']) * val + p['epsilon'] * (2 * np.random.random_sample() - 1)
            if m == 0 or (game.whiteToMove and epsVal > bestVal) or (not game.whiteToMove and epsVal < bestVal):
                bestVal = epsVal
                bestMove = copy.deepcopy(legalMoves[m])
                bestInput = (np.copy(netInputs[0]), np.copy(netInputs[1]))
                
            captCount += legalMoves[m].isCapture(game)           

        game.doMove(copy.deepcopy(bestMove))
        bestMoves.append(bestMove)
        bestInputs.append(bestInput)
        captures.append(captCount)

    gBuffer = p['gBuffer']
    tDepth = p['tDepth']
    rDepth = p['rDepth']
    #   Snip off game end as specified by gBuffer (if possible)
    if len(bestMoves) > gBuffer:
        if gBuffer > 0:
            bestMoves = bestMoves[:-1*gBuffer]
    else:
        print("Warning: game was shorter than gBuffer. Ignoring gBuffer choice and proceeding...")

    #   Randomly determine the location of the base node, considering
    #   tBuffer, tDepth, and gBuffer
    intervalLen = len(bestMoves) - 2 * gBuffer - rDepth - tDepth + 1
    if intervalLen <= 0 and len(bestMoves) - tDepth - rDepth > 0:
        print("Warning: game was not long enough to obey all user choices. Proceeding with gBuffer = 0")
        intervalLen += 2 * gBuffer
        gBuffer = 0
    elif intervalLen <= 0:
        print("Fatal error: user specifications cannot be met even with gBuffer=0.")

    #   Now select the index randomly with chance proportional to the number of captures available
    #   at that index
    cumDistr = [captures[gBuffer]]
    for i in range(intervalLen-1):
        cumDistr.append(captures[gBuffer+i] + cumDistr[i-1])

    r = np.random.randint(cumDistr[-1],size=1)
    #   Move "right" in the interval [0,cumDistr[-1]] and count how many
    #   cumDistr elements you pass before reaching the random val
    d = 0
    i = -1
    while d < r:
        i += 1
        d = cumDistr[i]
    gStart = i

    #   For debugging
    print("Game randomly chosen to start at move", i, "where there are", captures[i], "captures.")

    #   Initialize game, gStart moves into the base game
    g = Game.Game()
    g.quiet = True
    for i in range(gStart):
        g.doMove(bestMoves[i])

    return (g, p)
         
def toWeightedRew(net, expR, actR, alpha, isWhite):
    #   Negate evals if black
    evals = [float(np.exp(i))*(isWhite * 2 - 1) for i in expR]
    evalSum = sum(evals)
    evals = [i / evalSum for i in evals]

    numMoves = len(evals)
    #   Take the desired linear combination of (reward weighted by probability of move) and (reward
    #   weighted by frequency of occurrence), where alpha represents competency of the
    #   network. The larger alpha is, the more reward is weighted by what the network
    #   thinks the associated move's probability is (rather than a complete guess- freq)
    former = sum([evals[i] * actR[i] for i in range(numMoves)])
    latter = sum(actR) / numMoves

    return alpha * former + (1 - alpha) * latter
        
    
