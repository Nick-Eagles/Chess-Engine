import Game
import Network
import network_helper
import misc
import board_helper

import numpy as np
from scipy.special import expit, logit

#################################################################################
#   Functions for sampling several legal moves (subsetting search tree)
#################################################################################

#   A function defining the policy choice for the net given the current game.
#   A subset of moves (min(breadth, len(legalMoves))) is selected by sampling
#   the legal moves available with probability equal to the softmax of the
#   NN evaluations, constraining each move to be distinct (w/o replacement).
#   "curiosity" is the coefficient to the exponential function, thus favoring
#   highly evaluated moves more when increased.
def sampleMovesSoft(net, game, breadth, curiosity, mateRew, reqMove=None):
    #   Get legal moves and NN evaluations on the positions that result from them
    moves = board_helper.getLegalMoves(game)
    fullMovesLen = len(moves)
    rPairs = [game.getReward(m, mateRew) for m in moves]
    evals = np.array([x + float(logit(net.feedForward(y))) for x,y in rPairs])
    if not game.whiteToMove:
        evals = -1 * evals

    temp = np.exp(curiosity * evals)
    probs = temp / np.sum(temp) # softmax of evals
    cumProbs = np.cumsum(probs)

    finalMoves = []
    if reqMove == None:
        numToChoose = min(breadth, len(moves))
    else:
        #   Get the index of the required move and make sure it can't be chosen
        #   (by setting its probability to 0, essentially)
        reqMoveInd = [i for i in range(len(moves)) if moves[i].equals(reqMove)][0]
        if reqMoveInd == 0:
            cumProbs[reqMove] = 0
        else:
            cumProbs[reqMove] = cumProbs[reqMoveInd-1]

        #   Add the required move manually
        finalMoves.append(moves.pop(reqMoveInd))
        numToChoose = min(breadth-1, len(moves))

    finalMoves += [moves[i] for i in misc.sampleCDF(cumProbs, numToChoose)]

    assert len(finalMoves) > 0 and len(finalMoves) <= breadth, len(finalMoves)
    return(finalMoves, fullMovesLen)

#   An alternative to "sampleMovesSoft". This uses what is intended to be a
#   multiple-move analogue of an epsilon-greedy decision policy: N distinct
#   moves are selected, each with an epsilon-greedy strategy, such that the
#   probability that the most highly evaluated move is one of the N chosen
#   moves is 1 - eps. Here N = min(breadth, len(getLegalMoves(game)))
def sampleMovesEG(net, game, breadth, eps, mateRew, reqMove=None):
    #   Get legal moves and NN evaluations on the positions that result from them
    moves = board_helper.getLegalMoves(game)
    fullMovesLen = len(moves)
    if breadth >= fullMovesLen:
        return (moves, fullMovesLen)
    
    rPairs = [game.getReward(m, mateRew) for m in moves]
    evals = np.array([x + float(logit(net.feedForward(y))) for x,y in rPairs])
    if not game.whiteToMove:
        evals = -1 * evals

    subMovesLen = min(breadth, fullMovesLen)
    #   This is the choice of epsilon such that if [subMovesLen] moves are chosen under
    #   an epsilon-greedy strategy, with each move constrained to be distinct, then the probability
    #   that none of those moves have the highest NN evaluation is eps.
    epsEffective = 1 - (fullMovesLen * eps / (fullMovesLen - subMovesLen))**(1/subMovesLen)
    inds = []
    remainInds = list(range(fullMovesLen))
    chooseBest = [p > epsEffective for p in np.random.uniform(size=subMovesLen)]
    for i in range(subMovesLen):
        if chooseBest[i]:
            temp = np.argmax(evals)
            evals[temp] = -2 * mateRew # which should be less than any eval
            inds.append(temp)
            remainInds.remove(temp)
        else:
            temp = remainInds.pop(np.random.randint(len(remainInds)))
            evals[temp] = -2 * mateRew
            inds.append(temp)

    return ([moves[i] for i in inds], fullMovesLen)

#################################################################################
#   Functions for picking a single legal move to play
#################################################################################

#   Return a move decision, given the current game, network, and choice of epsilon.
#   The decision is meant to be very "human" in nature: gaussian noise is added to
#   the evaluations, matching their mean and variance, and the best combination of
#   evaluation and noise is chosen. Epsilon scales how noisy the decision is.
def getBestMoveHuman(game, legalMoves, net, p):
    eps = p['epsilon']
    if eps == 1:
        #   Shortcut for completely random move choice
        return legalMoves[np.random.randint(len(legalMoves))]
    else:
        #   Get the expected future reward
        vals = np.zeros(len(legalMoves), dtype=np.float32)
        for i, m in enumerate(legalMoves):
            rTuple = game.getReward(m, p['mateReward'])
            vals[i] = rTuple[0] + float(logit(net.feedForward(rTuple[1])))

        #   Return the best move as the legal move maximizing the linear combination of:
        #       1. The expected future reward vector
        #       2. A noise vector matching the first 2 moments of the reward vector
        noise = np.random.normal(np.mean(vals), np.std(vals), vals.shape[0])
        if game.whiteToMove:
            bestMove = legalMoves[np.argmax((1 - eps) * vals + eps * noise)]
        else:
            bestMove = legalMoves[np.argmin((1 - eps) * vals + eps * noise)]
            
        return bestMove

#   Return a move decision, given the current game, network, and choice of epsilon.
#   This is meant to be a faster alternative to getBestMoveHuman. The move is simply
#   chosen via an epsilon-greedy strategy.
def getBestMoveEG(game, legalMoves, net, p):
    eps = p['epsilon']
    if eps == 1 or np.random.uniform() < eps:
        #   Shortcut for completely random move choice
        return legalMoves[np.random.randint(len(legalMoves))]
    else:
        #   Get the expected future reward
        vals = np.zeros(len(legalMoves), dtype=np.float32)
        for i, m in enumerate(legalMoves):
            rTuple = game.getReward(m, p['mateReward'])
            vals[i] = rTuple[0] + float(logit(net.feedForward(rTuple[1])))

        if game.whiteToMove:
            return legalMoves[np.argmax(vals)]
        else:
            return legalMoves[np.argmin(vals)]
