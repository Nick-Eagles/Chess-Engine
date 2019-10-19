import Game
import board_helper
import input_handling
import policy

import numpy as np
from scipy.special import expit, logit
from multiprocessing import Pool
import os

def generateExamples(net, p):
    game = Game.Game()
    step = 0
    rewards = [[]]  # Each element is a list representing rewards for moves during one game
    NN_vecs = [] #  This follows 'rewards' but is flattened, and trimmed at the end of each game
    game_results = []

    #   Continue taking actions and receiving rewards (start a new game if necessary)
    while step < p['maxSteps']:
        while (game.gameResult == 17 and step < p['maxSteps']):
            #   Get best move (epsilon-greedy policy)
            legalMoves = board_helper.getLegalMoves(game)
            bestMove = policy.getBestMoveEG(game, legalMoves, net, p['epsGreedy'], p['mateReward'])

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

#   Simple wrapper allowing multithreading on functions w/ several parameters
def runThread(inTuple):
    return generateExamples(inTuple[0], inTuple[1])

def aync_q_learn(net):
    p = input_handling.readConfig(3)
    inList = [(net, p) for i in range(p['baseBreadth'])]
    if p['mode'] >= 2:
        print(os.cpu_count(), "cores available.")

    print("Performing asynchronous Q-learning (" + str(p['baseBreadth']) + " tasks)...")
    pool = Pool()
    thread_data = pool.map_async(runThread, inList).get()
    pool.close()

    tData = []
    for data in thread_data:
        tData += data
    print("Done. Generated " + str(len(tData)) + " training examples.")

    if p['mode'] >= 1:
        print("Determining certainty of network on the generated examples...")
        getCertainty(net, tData)

    return tData

def getCertainty(net, data):
    #   Form vectors of expected and actual rewards received
    expRew = logit(np.array([net.feedForward(data[i][0]) for i in range(len(data)) if i % 2 == 0]).flatten())
    actRew = logit(np.array([data[i][1] for i in range(len(data)) if i % 2 == 0]).flatten())

    #   Normalized dot product of expected and actual reward vectors
    certainty = np.dot(expRew, actRew) / (np.linalg.norm(expRew) * np.linalg.norm(actRew))

    print("Certainty of network on", int(len(data)/2), "examples:", certainty)
