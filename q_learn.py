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
        
    game = Game.Game()
    step = 0
    rewards = [[]]  # Each element is a list representing rewards for moves during one game
    NN_vecs = [] #  This follows 'rewards' but is flattened, and trimmed at the end of each game
    game_results = []

    #   Continue taking actions and receiving rewards (start a new game if necessary)
    while step < p['maxSteps']:
        while (game.gameResult == 17 and step < p['maxSteps']):
            bestMove = policy.getBestMoveTreeEG(net, game, p)

            #   Append the reward received from the best move
            rewards[len(game_results)].append(game.getReward(bestMove, p['mateReward'])[0])

            #   Perform move
            NN_vecs.append(game.toNN_vecs())
            game.doMove(bestMove)
            step += 1

        #   Trim end of game (or end of state sequence, more generally)
        halfMoveNum = (not game.whiteToMove) + 2*(game.moveNum - 1)
        if game.gameResult == 17:
            NN_vecs = NN_vecs[:(-1 * min(halfMoveNum, p['rDepthMin']))]
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
            trim_len = p['rDepthMin']
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

    data = [[],[],[]]
    for i, r in enumerate(rSeqFlat):
        data[0].append((NN_vecs[i][0], expit(r)))
        data[0].append((NN_vecs[i][1], expit(-1 * r)))
        if len(NN_vecs[i]) > 2:
            data[1].append((NN_vecs[i][2], expit(r)))
            data[1].append((NN_vecs[i][3], expit(-1 * r)))
            if len(NN_vecs[i]) == 16:
                for j in range(4, 10):
                    data[2].append((NN_vecs[i][j], expit(r)))
                for j in range(10, 16):
                    data[2].append((NN_vecs[i][j], expit(-1 * r)))

    board_helper.verify_data(data, False)
    
    return data

def async_q_learn(net):
    p = input_handling.readConfig(1)
    p.update(input_handling.readConfig(3))
    
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
    tData = [[],[],[]]
    for data in thread_data:
        for i in range(3):
            tData[i] += data[i]

    board_helper.verify_data(tData, withMates=False)

    if p['mode'] >= 2:
        elapsed = round(time.time() - start_time, 2)
        print("Done in", elapsed, "seconds. Generated " + str(sum([len(x) for x in tData])) + " training examples.\n")
    else:
        
        print("Done. Generated " + str(sum([len(x) for x in tData])) + " training examples.\n")

    print("Determining certainty of network on the generated examples...")
    getCertainty(net, tData, p)

    return tData

def getCertainty(net, data, p):
    #   Get only the originally generated examples (do not include augmented data)
    origData = [data[0][i] for i in range(len(data[0])) if i % 2 == 0] + \
               [data[1][i] for i in range(len(data[1])) if i % 4 == 0] + \
               [data[2][i] for i in range(len(data[2])) if i % 16 == 0]
    #   Form vectors of expected and actual rewards received
    expRew = logit(np.array([net.feedForward(x[0]) for x in origData]).flatten())
    actRew = logit(np.array([x[1] for x in origData]).flatten())

    #   Normalized dot product of expected and actual reward vectors
    certainty = np.dot(expRew, actRew) / (np.linalg.norm(expRew) * np.linalg.norm(actRew))
    if p['epsGreedy'] < 0.5:
        #   0.5 is an arbitrarily established cutoff to prevent catastrophic
        #   amplification of 'noise' in estimating true certainty
        certainty /= 1 - p['epsGreedy']

    #   Adjust certainty (an exponentially weighted moving average)
    net.certaintyRate = net.certaintyRate * p['persist'] + (certainty - net.certainty) * (1 - p['persist'])
    net.certainty = net.certainty * p['persist'] + certainty * (1 - p['persist'])
    
    if p['mode'] >= 1:
        print("Certainty of network on", len(origData), "examples:", round(certainty, 5))
        print("Moving certainty:", round(net.certainty, 5))
        print("Moving rate of certainty change:", round(net.certaintyRate, 5), "\n")
