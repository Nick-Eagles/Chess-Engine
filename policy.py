import Game
import Network
import network_helper
import misc
import board_helper
import Traversal

import numpy as np
from scipy.special import expit, logit
import os
from multiprocessing import Pool

#################################################################################
#   Functions for sampling several legal moves (subsetting search tree)
#################################################################################

#   A function defining the policy choice for the net given the current game.
#   A subset of moves (min(breadth, len(legalMoves))) is selected by sampling
#   the legal moves available with probability equal to the softmax of the
#   NN evaluations, constraining each move to be distinct (w/o replacement).
#   "curiosity" is the coefficient to the exponential function, thus favoring
#   highly evaluated moves more when increased.
def sampleMovesSoft(net, game, p, reqMove=None):
    #   Get legal moves and NN evaluations on the positions that result from them
    moves = board_helper.getLegalMoves(game)
    fullMovesLen = len(moves)
    rPairs = [game.getReward(m, p['mateReward']) for m in moves]
    evals = np.array([x + float(logit(net.feedForward(y))) for x,y in rPairs])
    if not game.whiteToMove:
        evals = -1 * evals

    temp = np.exp(p['curiosity'] * evals)
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

    assert len(finalMoves) > 0 and len(finalMoves) <= p['breadth'], len(finalMoves)
    return(finalMoves, fullMovesLen)

#   An alternative to "sampleMovesSoft". This uses what is intended to be a
#   multiple-move analogue of an epsilon-greedy decision policy: N distinct
#   moves are selected, each with an epsilon-greedy strategy, such that the
#   probability that the most highly evaluated move is one of the N chosen
#   moves is 1 - eps. Here N = min(breadth, len(getLegalMoves(game)))
def sampleMovesEG(net, game, p, reqMove=None):
    moves = board_helper.getLegalMoves(game)

    #   Simply choose all moves if the breadth spans this far
    fullMovesLen = len(moves)
    assert fullMovesLen > 0
    if p['breadth'] >= fullMovesLen:
        return (moves, fullMovesLen)
    
    #   Determine which moves should be chosen randomly
    subMovesLen = min(p['breadth'], fullMovesLen)
    #   This is the choice of epsilon such that if [subMovesLen] moves are chosen under
    #   an epsilon-greedy strategy, with each move constrained to be distinct, then the probability
    #   that none of those moves have the highest NN evaluation is eps.
    epsEffective = 1 - (fullMovesLen * p['epsGreedy'] / (fullMovesLen - subMovesLen))**(1/subMovesLen)
    inds = []
    remainInds = list(range(fullMovesLen))
    chooseBest = [x > epsEffective for x in np.random.uniform(size=subMovesLen)]
    
    #   If all moves are to be chosen randomly, don't even compute their evals
    if not any(chooseBest):
        for i in range(subMovesLen):
            temp = remainInds.pop(np.random.randint(len(remainInds)))
            inds.append(temp)
    else:
        #   Compute evaluations on each move
        rPairs = [game.getReward(m, p['mateReward']) for m in moves]
        evals = np.array([x + float(logit(net.feedForward(y))) for x,y in rPairs])
        if not game.whiteToMove:
            evals = -1 * evals

        if p['epsGreedy'] == 0:
            inds = misc.topN(evals, p['breadth'])
        else:
            #   Select distinct moves via an epsilon-greedy policy
            for i in range(subMovesLen):
                if chooseBest[i]:
                    temp = np.argmax(evals)
                    assert min(evals) >= -2 * p['mateReward'], min(evals)
                    evals[temp] = -2 * p['mateReward'] # which should be less than any eval
                    inds.append(temp)
                    remainInds.remove(temp)
                else:
                    temp = remainInds.pop(np.random.randint(len(remainInds)))
                    evals[temp] = -2 * p['mateReward']
                    inds.append(temp)

    return ([moves[i] for i in inds], fullMovesLen)

#################################################################################
#   Functions for picking a single legal move to play
#################################################################################

#   Return a move decision, given the current game, network, and choice of epsilon.
#   The decision is meant to be very "human" in nature: gaussian noise is added to
#   the evaluations, matching their mean and variance, and the best combination of
#   evaluation and noise is chosen. Epsilon scales how noisy the decision is.
def getBestMoveHuman(net, game, p):
    legalMoves = board_helper.getLegalMoves(game)
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
def getBestMoveEG(net, game, p):
    legalMoves = board_helper.getLegalMoves(game)
    
    if p['epsGreedy'] == 1 or np.random.uniform() < p['epsGreedy']:
        #   Shortcut for completely random move choice
        return legalMoves[np.random.randint(len(legalMoves))]
    else:
        #   Get the expected future reward
        vals = np.zeros(len(legalMoves))
        for i, m in enumerate(legalMoves):
            rTuple = game.getReward(m, p['mateReward'])
            vals[i] = rTuple[0] + p['alpha'] * float(logit(net.feedForward(rTuple[1])))

        if game.whiteToMove:
            return legalMoves[np.argmax(vals)]
        else:
            return legalMoves[np.argmin(vals)]

def per_thread_job(trav_obj):
    trav_obj.traverse()
    return trav_obj

def getBestMoveTreeEG(net, game, p, pool=None):
    legalMoves = board_helper.getLegalMoves(game)

    #   Traversals are started at positions resulting from testing moves from
    #   the current position; this test constitutes a step of depth
    p_copy = p.copy()
    if p['rDepth'] == 0:
        p_copy['tDepth'] -= 1
    else:
        p_copy['rDepth'] -= 1
    p = p_copy
    
    if p['epsGreedy'] == 1 or np.random.uniform() < p['epsGreedy']:
        #   Shortcut for completely random move choice
        return legalMoves[np.random.randint(len(legalMoves))]
    else:
        #   Get NN evaluations on each possible move
        evals = np.zeros(len(legalMoves))
        rTemp = np.zeros(len(legalMoves))
        for i, m in enumerate(legalMoves):
            rTuple = game.getReward(m, p['mateReward'])
            evals[i] = rTuple[0] + p['alpha'] * float(logit(net.feedForward(rTuple[1])))
            rTemp[i] = rTuple[0]
        if not game.whiteToMove:
            evals *= -1

        best_inds = misc.topN(evals, p['breadth'])
        rTemp = rTemp[np.array(best_inds)]

        #   Get best move from a tree search
        p_copy = p.copy()
        p_copy['epsGreedy'] = 0
        if not (pool == None):
            realBreadth = min(os.cpu_count(), len(legalMoves))
            certainty = 1 - (len(legalMoves) - realBreadth) * (1 - p['alpha'])**realBreadth / len(legalMoves)

            trav_objs = []
            for i, m in enumerate([legalMoves[ind] for ind in best_inds]):
                g = game.copy()
                g.quiet = True
                g.doMove(m)
                trav_objs.append(Traversal.Traversal(g, net, p_copy, isBase=False, collectData=False))

            #   Perform the traversals in parallel to return the index of the first
            #   move from this position of the most rewarding move sequence explored
            res_objs = pool.map(per_thread_job, trav_objs)

            #   Here we check for the presence of a mate in 1, which should override the normal scaling
            #   by certainty (since we are guaranteed we found the optimal move, mate, from this node)
            baseRs = np.array([ob.baseR for ob in res_objs])
            temp_bools = np.absolute(baseRs) == p['mateReward']
            if any(temp_bools):
                baseRs[temp_bools] /= certainty    
            rTemp += certainty * baseRs
        else:  
            realBreadth = min(p['breadth'], len(legalMoves))
            certainty = 1 - (len(legalMoves) - realBreadth) * (1 - p['alpha'])**realBreadth / len(legalMoves)
            for i, m in enumerate([legalMoves[ind] for ind in best_inds]):
                g = game.copy()
                g.quiet = True
                g.doMove(m)
                    
                trav = Traversal.Traversal(g, net, p, isBase=False, collectData=False)
                trav.traverse()
                rTemp[i] += certainty * trav.baseR

        if game.whiteToMove:
            bestMove = legalMoves[best_inds[np.argmax(rTemp)]]
        else:
            bestMove = legalMoves[best_inds[np.argmin(rTemp)]]

        return bestMove
