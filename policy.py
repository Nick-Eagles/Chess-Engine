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
def sampleMovesSoft(net, game, p):
    #   Get legal moves and NN evaluations on the positions that result from them
    moves = board_helper.getLegalMoves(game)
    fullMovesLen = len(moves)
    evals = getEvals(moves, net, game, p)

    temp = np.exp(p['curiosity'] * evals)
    probs = temp / np.sum(temp) # softmax of evals
    cumProbs = np.cumsum(probs)

    finalMoves = [moves[i] for i in misc.sampleCDF(cumProbs, min(breadth, len(moves)))]

    assert len(finalMoves) > 0 and len(finalMoves) <= p['breadth'], len(finalMoves)
    return(finalMoves, fullMovesLen)

#   An alternative to "sampleMovesSoft". This uses what is intended to be a
#   multiple-move analogue of an epsilon-greedy decision policy: N distinct
#   moves are selected, each with an epsilon-greedy strategy, such that the
#   probability that the most highly evaluated move is one of the N chosen
#   moves is 1 - eps. Here N = min(breadth, len(getLegalMoves(game)))
def sampleMovesEG(net, game, p):
    moves = board_helper.getLegalMoves(game)

    #   Simply choose all moves if the breadth spans this far
    fullMovesLen = len(moves)
    assert fullMovesLen > 0
    if p['breadth'] >= fullMovesLen:
        return (moves, fullMovesLen)

    #   More efficiently handle a trivial case
    if p['epsSearch'] == 0:
        evals = getEvals(moves, net, game, p)
        return ([moves[i] for i in misc.topN(evals, p['breadth'])], fullMovesLen)
    
    #   Determine which moves should be chosen randomly
    subMovesLen = min(p['breadth'], fullMovesLen)
    #   This is the choice of epsilon such that if [subMovesLen] moves are chosen under
    #   an epsilon-greedy strategy, with each move constrained to be distinct, then the probability
    #   that none of those moves have the highest NN evaluation is eps.
    epsEffective = (fullMovesLen * p['epsSearch'] / (fullMovesLen - subMovesLen))**(1/subMovesLen)
    inds = []
    remainInds = list(range(fullMovesLen))
    chooseBest = [x > epsEffective for x in np.random.uniform(size=subMovesLen)]
    
    #   If all moves are to be chosen randomly, don't even compute their evals
    if not any(chooseBest):
        for i in range(subMovesLen):
            temp = remainInds.pop(np.random.randint(len(remainInds)))
            inds.append(temp)
    else:
        evals = getEvals(moves, net, game, p)

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
        #   Return the best move as the legal move maximizing the linear combination of:
        #       1. The expected future reward vector
        #       2. A noise vector matching the first 2 moments of the reward vector
        vals = getEvals(moves, net, game, p)
        noise = np.random.normal(np.mean(vals), np.std(vals), vals.shape[0])
        
        bestMove = legalMoves[np.argmax((1 - eps) * vals + eps * noise)]
            
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
        vals = getEvals(moves, net, game, p)

        return legalMoves[np.argmax(vals)]

def per_thread_job(trav_obj):
    trav_obj.traverse()
    return trav_obj

def getBestMoveTreeEG(net, game, p, pool=None):
    if np.random.uniform() < p['epsGreedy']:
        legalMoves = board_helper.getLegalMoves(game)
        return legalMoves[np.random.randint(len(legalMoves))]
    else:
        #   Traversals are started at positions resulting from testing moves from
        #   the current position; this test constitutes a step of depth
        p_copy = p.copy()
        p_copy['depth'] -= 1
        p = p_copy
        
        moves, fullMovesLen = sampleMovesEG(net, game, p)
        #certainty = 1 - (fullMovesLen - len(moves)) * (1 - p['alpha'])**len(moves) / fullMovesLen
        
        rTemp = np.zeros(len(moves))
        if pool != None:
            trav_objs = []
            for i, m in enumerate(moves):
                rTemp[i] = game.getReward(m, p['mateReward'], True)[0]

                g = game.copy()
                g.quiet = True
                g.doMove(m)
                trav_objs.append(Traversal.Traversal(g, net, p))

            #   Perform the traversals in parallel to return the index of the first
            #   move from this position of the most rewarding move sequence explored
            res_objs = pool.map(per_thread_job, trav_objs)

            #   Here we check for the presence of a mate in 1, which should override the normal decay
            #   by gamma (since we are guaranteed we found the optimal move, mate, from this node)
            baseRs = np.array([ob.baseR for ob in res_objs])
            temp_bools = np.absolute(baseRs) == p['mateReward']
            if any(temp_bools):
                baseRs[temp_bools] = baseRs[temp_bools] / p['gamma_exec']
                #baseRs[temp_bools] = baseRs[temp_bools] / certainty
            rTemp += p['gamma_exec'] * baseRs
            #rTemp += certainty * baseRs
        else:  
            for i, m in enumerate(moves):
                g = game.copy()
                g.quiet = True
                g.doMove(m)
                        
                trav = Traversal.Traversal(g, net, p)
                trav.traverse()
                rTemp[i] = game.getReward(m, p['mateReward'], True)[0] + p['gamma_exec'] * trav.baseR
                #rTemp[i] = game.getReward(m, p['mateReward'], True)[0] + certainty * trav.baseR

        if game.whiteToMove:
            bestMove = moves[np.argmax(rTemp)]
        else:
            bestMove = moves[np.argmin(rTemp)]

        return bestMove

#   A helper function to compute depth-1 evaluations of a list of moves. Returns
#   a list the same length as 'moves', with each component being the sum of the
#   immediate reward for performing the respective move and a scaled
#   NN-evaluation of the resulting position. The NN evaluation is computed only
#   if net.certainty is positive, in which case net.certainty is the scalar to
#   the NN-evaluation. Note that evals are flipped: larger values match to
#   better moves for the current player!
def getEvals(moves, net, game, p):
    #   Compute NN evaluations on each move if certainty is positive
    evals = np.zeros(len(moves))
    if net.certainty > p['minCertainty']:
        scalar = p['gamma_exec'] * net.certainty / p['gamma']
        for i, m in enumerate(moves):
            temp = game.getReward(m, p['mateReward'])
            evals[i] = temp[0] + scalar * float(logit(net.feedForward(temp[1])))
    else:
        evals = np.array([game.getReward(m, p['mateReward'], True)[0] for m in moves])

        #   If evals are not all unique, add a tiny amount of noise to eliminate
        #   bias for moves occuring earlier in the list of legal moves
        if not misc.is_unique(evals):
            noise = np.random.normal(scale=0.0001, size=evals.shape)
            evals = evals + noise
    
    if not game.whiteToMove:
        evals = -1 * evals

    assert evals.shape == (len(moves),), evals.shape
    return evals
