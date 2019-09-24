import Game
import Network
import network_helper
import misc
import board_helper

import numpy as np
from scipy.special import expit, logit

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

    subsMovesLen = min(breadth, fullMovesLen)
    #   This is the choice of epsilon such that if [subMovesLen] moves are chosen under
    #   an epsilon-greedy strategy, with each move constrained to be distinct, then the probability
    #   that none of those moves have the highest NN evaluation is eps.
    epsEffective = 1 - (fullMovesLen * eps / (fullMovesLen - subMovesLen))**(1/subMovesLen)
    inds = []
    remainInds = list(range(subMovesLen))
    chooseBest = [p > epsEffective for p in np.random.unif(size=subMovesLen)]
    for i in range(subMovesLen):
        if chooseBest[i]:
            temp = np.argmax(evals)
            evals[temp] = -2 * mateRew # which should be less than any eval
        else:
            temp = np.random.randint(len(remainInds))
        inds.append(remainInds.pop(temp))

    return ([moves[i] for i in inds], fullMovesLen)
