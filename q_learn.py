import Game
import board_helper
import input_handling
import policy
import misc
import Traversal

import numpy as np
from scipy.special import logit, expit
import os
import time
import tensorflow as tf

def generateExamples(net, p):
    if p['mode'] >= 3:
        np.random.seed(0)
    else:
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
            rewards[len(game_results)].append(game.getReward(bestMove, p['mateReward'], True)[0])

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
        num_examples = len(NN_vecs[i])

        #   Based on the number of examples for this particular position,
        #   assign the examples to the appropriate data buffer
        if num_examples == 2:
            buffer_index = 0
        elif num_examples == 4:
            buffer_index = 1
        elif num_examples == 16:
            buffer_index = 2
        else:
            sys.exit("Received an invalid number of augmented positions" + \
                     "associated with one example: " + str(num_examples))

        #   Add the data to the correct buffer
        data[buffer_index] += [(NN_vecs[i][j], index_to_label(j, r))
                               for j in range(len(NN_vecs[i]))]

    board_helper.verify_data(data, p, 3)
    
    return data


#   An 'ad hoc' function: take any particular element of 'NN_vecs' as produced
#   by 'generateExamples', and call it 'NN_vec_mirrors'. This is a list of
#   inputs to the NN, where NN_vec_mirrors[0] is the original position, and
#   NN_vec_mirrors[j] are augmented variations of this position for j >= 1. Let
#   the parameter 'r' be the raw reward associated with NN_vec_mirrors[0]. If
#   we are considering the position NN_vec_mirrors[i], this function returns the
#   appropriate label for this position (a tf.Tensor of shape [1,1]).
def index_to_label(i, r):
    if i == 0 or i == 2 or (i >= 4 and i < 10):
        return tf.constant(expit(r), shape=[1,1], dtype=tf.float32)
    else:
        return tf.constant(expit(-1 * r), shape=[1,1], dtype=tf.float32)


def async_q_learn(net):
    p = input_handling.readConfig(1)
    p.update(input_handling.readConfig(3))

    print("Performing Q-learning (" + str(p['baseBreadth']) + " tasks)...")
    if p['mode'] >= 2:
        start_time = time.time()

    #   Run asynchronous data generation
    data = [[], [], []]
    for i in range(p['baseBreadth']):
        this_data = generateExamples(net, p)
        for j in range(3):
            data[j] += this_data[j]

    board_helper.verify_data(data, p, 3)

    if p['mode'] >= 2:
        elapsed = round(time.time() - start_time, 2)
        print("Done in", elapsed, "seconds. Generated " + str(sum([len(x) for x in data])) + " training examples.\n")
    else:
        print("Done. Generated " + str(sum([len(x) for x in data])) + " training examples.\n")

    print("Determining certainty of network on the generated examples...")
    getCertainty(net, data, p)

    return data

def getCertainty(net, data, p):
    #   Get only the originally generated examples (do not include augmented data)
    origData = [data[0][i] for i in range(len(data[0])) if i % 2 == 0] + \
               [data[1][i] for i in range(len(data[1])) if i % 4 == 0] + \
               [data[2][i] for i in range(len(data[2])) if i % 16 == 0]
    #   Form vectors of expected and actual rewards received
    inputs = tf.stack([tf.reshape(x[0], [839]) for x in origData], axis=0)
    expRew = logit(net(inputs, training=False)).flatten()
    actRew = logit(tf.stack([tf.reshape(x[1], [1]) for x in origData], axis=0)).flatten()

    #   Normalized dot product of expected and actual reward vectors
    actNorm = np.linalg.norm(actRew)
    if round(float(actNorm), 5) == 0:
        certainty = 0
    else:
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

        if p['mode'] >= 2:
            #   Print first 2 moments of expected and actual rewards
            print("Mean and var of expected reward: ", round(np.mean(expRew), 4), "; ", round(np.var(expRew), 4), sep="")
            print("Mean and var of actual reward: ", round(np.mean(actRew), 4), "; ", round(np.var(actRew), 4), sep="")
