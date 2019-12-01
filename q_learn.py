import Game
import board_helper
import input_handling
import policy
import misc
import Traversal

import numpy as np
from scipy.special import expit, logit
from multiprocessing import Pool
import os
import time

def generateExamples(net, p):
    np.random.seed()
    p_copy = p.copy()
    if p['rDepth'] == 0:
        p_copy['tDepth'] -= 1
    else:
        p_copy['rDepth'] -= 1
    p = p_copy
        
    game = Game.Game()
    step = 0
    rewards = [[]]  # Each element is a list representing rewards for moves during one game
    NN_vecs = [] #  This follows 'rewards' but is flattened, and trimmed at the end of each game
    game_results = []

    #   Continue taking actions and receiving rewards (start a new game if necessary)
    while step < p['maxSteps']:
        while (game.gameResult == 17 and step < p['maxSteps']):
            legalMoves = board_helper.getLegalMoves(game)
            
            #   Get NN evaluations on each possible move
            evals = np.zeros(len(legalMoves))
            rTemp = np.zeros(len(legalMoves))
            for i, m in enumerate(legalMoves):
                rTuple = game.getReward(m, p['mateReward'])
                evals[i] = rTuple[0] + float(logit(net.feedForward(rTuple[1])))
                rTemp[i] = rTuple[0]

            best_inds = misc.topN(evals, p['breadth'])
            rTemp = rTemp[np.array(best_inds)]

            #   Get best move 
            rTemp = np.full(min(p['breadth'], len(legalMoves)), rTuple[0])
            for i, m in enumerate([legalMoves[ind] for ind in best_inds]):
                g = game.copy()
                g.quiet = True
                g.doMove(m)
                
                trav = Traversal.Traversal(g, net, p, isBase=False, collectData=False, best=True)
                trav.traverse()
                rTemp[i] += p['gamma'] * trav.baseR

            bestMove = legalMoves[best_inds[np.argmax(rTemp)]]

            #   Append the reward received from the best move
            rewards[len(game_results)].append(game.getReward(bestMove, p['mateReward'])[0])

            #   Perform move
            NN_vecs.append(game.toNN_vecs())
            game.doMove(bestMove)
            step += 1

        #   Trim end of game (or end of state sequence, more generally)
        halfMoveNum = (not game.whiteToMove) + 2*(game.moveNum - 1)
        if game.gameResult == 17:
            NN_vecs = NN_vecs[:(-1 * min(halfMoveNum, p['rDepth']))]
        else:
            NN_vecs.pop()

        game_results.append(game.gameResult)

        #   Begin a new game if one ended but we still have more steps to take
        if step < p['maxSteps']:
            game = Game.Game()
            rewards.append([])


    #   "Unravel" reward sequences: assign each state an expected cumulative reward value,
    #   with minimum depth given by p['rDepth']
    rSeqFlat = []
    for i, rSeqIn in enumerate(rewards):
        #   Determine how many states to trim at the end of the game (to not use
        #   as training data)
        if game_results[i] == 17:
            trim_len = p['rDepth']
        else:
            trim_len = 1

        #   If there are enough states, unravel the rewards for each usable state
        if len(rSeqIn) > trim_len:
            rSeqOut = np.zeros(len(rSeqIn) - trim_len)

            #   Take the last reward and propagate backward through the game
            rCum = rSeqIn[-1]
            for j in range(1,len(rSeqIn)):
                rCum = p['gamma'] * rCum + rSeqIn[-1 - j]
                if j >= trim_len:
                    rSeqOut[trim_len - 1 - j] = rCum

            rSeqFlat += rSeqOut.tolist()

    #   From the list of tuples "NN_vecs" and the list rSeqFlat, form training examples in
    #   the format expected by the rest of the program (such as in main.py)
    debug_str = "len(rewards): " + str(len(rewards)) + "\ngameNum: " + str(len(game_results)) \
                + "\nlen(NN_vecs): " + str(len(NN_vecs)) + "\nlen(rSeqFlat): " + str(len(rSeqFlat)) \
                + "\nstep: " + str(step) + "\nLast game result: " + str(game.gameResult) \
                + "\nLast game moveNum: " + str(game.moveNum)
    assert len(NN_vecs) == len(rSeqFlat), debug_str

    data = []
    for i, r in enumerate(rSeqFlat):
        data.append((NN_vecs[i][0], expit(r)))
        data.append((NN_vecs[i][1], expit(-1 * r)))

    return data

def async_q_learn(net):
    p = input_handling.readConfig(3)
    p.update(input_handling.readConfig(1))
    
    if p['mode'] >= 2:
        print(os.cpu_count(), "cores available.")

    print("Performing asynchronous Q-learning (" + str(p['baseBreadth']) + " tasks)...")
    if p['mode'] >= 2:
        start_time = time.time()

    #   Run asynchronous data generation
    pool = Pool()
    inList = [(net, p) for i in range(p['baseBreadth'])]
    thread_data = pool.starmap_async(generateExamples, inList).get()
    pool.close()

    #   Collect each process's results (data) into a single list
    tData = []
    for data in thread_data:
        tData += data

    if p['mode'] >= 2:
        elapsed = round(time.time() - start_time, 2)
        print("Done in", elapsed, "seconds. Generated " + str(len(tData)) + " training examples.\n")
    else:
        print("Done. Generated " + str(len(tData)) + " training examples.\n")

    if p['mode'] >= 1:
        print("Determining certainty of network on the generated examples...")
        getCertainty(net, tData, p)

    return tData

def getCertainty(net, data, p):
    #   Form vectors of expected and actual rewards received
    expRew = logit(np.array([net.feedForward(data[i][0]) for i in range(len(data)) if i % 2 == 0]).flatten())
    actRew = logit(np.array([data[i][1] for i in range(len(data)) if i % 2 == 0]).flatten())

    #   Normalized dot product of expected and actual reward vectors
    certainty = np.dot(expRew, actRew) / (np.linalg.norm(expRew) * np.linalg.norm(actRew))

    #   Adjust certainty (an exponentially weighted moving average)
    net.certaintyRate = net.certaintyRate * p['persist'] + (certainty - net.certainty) * (1 - p['persist'])
    net.certainty = net.certainty * p['persist'] + certainty * (1 - p['persist'])
    
    if p['mode'] >= 1:
        print("Certainty of network on", int(len(data)/2), "examples:", round(certainty, 5))
        print("Moving certainty:", round(net.certainty, 5))
        print("Moving rate of certainty change:", round(net.certaintyRate, 5), "\n")
