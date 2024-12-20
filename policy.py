import numpy as np
from scipy.special import logit
import tensorflow as tf

import policy_net
import misc
import board_helper
import Traversal

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
    #   Get legal moves and NN evaluations on the positions that result from
    #   them
    moves = board_helper.getLegalMoves(game)
    fullMovesLen = len(moves)
    evals = p['evalFun'](moves, net, game, p)

    temp = np.exp(p['curiosity'] * evals)
    probs = temp / np.sum(temp) # softmax of evals
    cumProbs = np.cumsum(probs)

    finalMoves = [
        moves[i] for i in misc.sampleCDF(cumProbs, min(breadth, len(moves)))
    ]

    assert len(finalMoves) > 0 and len(finalMoves) <= p['breadth'], \
           len(finalMoves)
    return (finalMoves, fullMovesLen)

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
        evals = p['evalFun'](moves, net, game, p)
        return ([moves[i] for i in misc.topN(evals, p['breadth'])], \
                fullMovesLen)
    
    #   Determine which moves should be chosen randomly
    subMovesLen = min(p['breadth'], fullMovesLen)
    #   This is the choice of epsilon such that if [subMovesLen] moves are
    #   chosen under an epsilon-greedy strategy, with each move constrained to
    #   be distinct, then the probability that none of those moves have the
    #   highest NN evaluation is eps.
    epsEffective = (fullMovesLen * p['epsSearch'] / (fullMovesLen - subMovesLen))**(1/subMovesLen)
    inds = []
    remainInds = list(range(fullMovesLen))
    numRandom = np.random.binomial(subMovesLen, epsEffective)
    
    #   If all moves are to be chosen randomly, don't even compute their evals
    if numRandom < subMovesLen:
        evals = p['evalFun'](moves, net, game, p)

        #   The moves to be chosen by best evaluation
        for i in range(subMovesLen - numRandom):
            temp = np.argmax(evals)
            assert min(evals) >= -2 * p['mateReward'], min(evals)
            evals[temp] = -2 * p['mateReward'] # which should be < any eval
            inds.append(temp)
            remainInds.remove(temp)

        #   The moves to randomly select
        for i in range(numRandom):
            temp = remainInds.pop(np.random.randint(len(remainInds)))
            evals[temp] = -2 * p['mateReward']
            inds.append(temp)
    else:
        #   Randomly select all moves
        for i in range(subMovesLen):
            temp = remainInds.pop(np.random.randint(len(remainInds)))
            inds.append(temp)
        

    return ([moves[i] for i in inds], fullMovesLen)

#################################################################################
#   Functions for picking a single legal move to play
#################################################################################

def returnBestMove(moves, vals, game, num_lines, interactive, best_lines=None, invert=True):
    if interactive:
        if game.whiteToMove or not invert:
            indices = misc.topN(vals, num_lines)
        else:
            indices = misc.topN(-1 * vals, num_lines)

        if best_lines is None:
            best_line = None
        else:
            best_line = best_lines[indices[0]]
        
        return (
            [moves[i] for i in indices],
            vals[indices],
            best_line
        )
    else:
        if game.whiteToMove or not invert:
            bestMove = moves[np.argmax(vals)]
        else:
            bestMove = moves[np.argmin(vals)]

        return bestMove

#   Return a move decision, given the current game, network, and choice of
#   epsilon. The decision is meant to be very "human" in nature: gaussian noise
#   is added to the evaluations, matching their mean and variance, and the best
#   combination of evaluation and noise is chosen. Epsilon scales how noisy the
#   decision is.
def getBestMoveHuman(net, game, p, interactive=False, num_lines=1):
    moves = board_helper.getLegalMoves(game)
    if p['epsGreedy'] == 1 or np.random.uniform() < p['epsGreedy']:
        #   Shortcut for completely random move choice
        return moves[np.random.randint(len(moves))]
    else:
        #   Return the best move as the legal move maximizing the linear
        #   combination of:
        #       1. The expected future reward vector
        #       2. A noise vector matching the first 2 moments of the reward
        #          vector
        vals = getEvals(moves, net, game, p)
        noise = np.random.normal(np.mean(vals), np.std(vals), vals.shape[0])

        vals = (1 - eps) * vals + eps * noise

        return returnBestMove(
            moves, vals, game, num_lines, interactive, None, True
        )


#   Return a move decision, given the current game, network, and choice of
#   epsilon. This is meant to be a faster alternative to getBestMoveHuman. The
#   move is simply chosen via an epsilon-greedy strategy.
def getBestMoveEG(net, game, p, interactive=False, num_lines=1):
    moves = board_helper.getLegalMoves(game)
    
    if p['epsGreedy'] == 1 or np.random.uniform() < p['epsGreedy']:
        #   Shortcut for completely random move choice
        return moves[np.random.randint(len(moves))]
    else:
        vals = getEvals(moves, net, game, p)

        return returnBestMove(
            moves, vals, game, num_lines, interactive, None, True
        )


def getBestMoveRawPolicy(net, game, p, interactive=False, num_lines=1):
    legalMoves = board_helper.getLegalMoves(game)
    
    if np.random.uniform() < p['epsGreedy']:
        return legalMoves[np.random.randint(len(legalMoves))]
    else:
        #   Compute a probability distribution across legal moves
        outputs = net(game.toNN_vecs(every=False)[0], training=False)[:3]
        probs = policy_net.AdjustPolicy(outputs, legalMoves, game.board)

        return returnBestMove(
            moves, probs, game, num_lines, interactive, None, False
        )
        

def getBestMoveTreeEG(net, game, p, interactive=False, num_lines=1):
    if np.random.uniform() < p['epsGreedy']:
        legalMoves = board_helper.getLegalMoves(game)
        return legalMoves[np.random.randint(len(legalMoves))]
    else:
        #   Traversals are started at positions resulting from testing moves
        #   from the current position; this test constitutes a step of depth
        p_copy = p.copy()
        p_copy['depth'] -= 1
        p = p_copy
        
        moves, fullMovesLen = sampleMovesEG(net, game, p)
        
        rTemp = np.zeros(len(moves))
        baseRs = np.zeros(len(moves))

        if interactive:
            bestLines = []
        else:
            bestLines = None

        for i, m in enumerate(moves):
            g = game.copy()
            g.quiet = True
                
            rTemp[i] = g.getReward(
                m, p['mateReward'], simple=True, copy=False
            )[0]
            t = Traversal.Traversal(g, net, p)
            t.traverse()
            baseRs[i] = t.baseR

            if interactive:
                #   Add a list of move names for the top line explored that
                #   started with this particular move "m"
                bestLines.append([m.getMoveName(game)] + t.bestLine)

        rTemp += p['gamma_exec'] * baseRs

        return returnBestMove(
            moves, rTemp, game, num_lines, interactive, bestLines, True
        )


#   A helper function to compute depth-1 evaluations of a list of moves. Returns
#   a list the same length as 'moves', with each component being the sum of the
#   immediate reward for performing the respective move and a scaled
#   NN-evaluation of the resulting position. Note that evals are flipped: larger
#   values match to better moves for the current player!
def getEvalsValue(moves, net, game, p):
    #   Compute NN evaluations on each move
    r_real = np.zeros(len(moves))
    net_inputs = []
    for i, m in enumerate(moves):
        r, vec = game.getReward(m, p['mateReward'])
        r_real[i] = r
        net_inputs.append(tf.reshape(vec, (839,)))

    #   Get the value output from the network, regardless of whether net is
    #   of type "policy-value" or "value"
    value = logit(net(tf.stack(net_inputs), training=False)).flatten()
    if isinstance(value, list):
        value = value[-1]
    
    evals = r_real + p['gamma_exec'] * value
    
    if not game.whiteToMove:
        evals = -1 * evals

    assert evals.shape == (len(moves),), evals.shape
    return evals


def getEvalsPolicy(moves, net, game, p):
    #   Compute a probability distribution across legal moves
    outputs = net(game.toNN_vecs(every=False)[0], training=False)[:2]
    evals = policy_net.AdjustPolicy(outputs, moves, game.board)
    
    return evals


def getEvalsHybrid(moves, net, game, p):
    #   Compute a probability distribution across legal moves
    outputs = net(game.toNN_vecs(every=False)[0], training=False)[:2]
    probs = policy_net.AdjustPolicy(outputs, moves, game.board)

    #   Get "empirical" rewards resulting from each move
    rewards = np.array(
        [game.getReward(m, p['mateReward'], simple=True)[0] for m in moves]
    )
    if not game.whiteToMove:
        rewards *= -1

    #   Compute a scalar in [0, 1], which is 0 if the most empirically
    #   rewarding move is as rewarding as checkmate, and 1 when all rewards
    #   are 0.
    scalar = max(
        (p['mateReward'] - np.max(np.abs(rewards))) / p['mateReward'],
        0
    )
    assert scalar >= 0 and scalar <= 1, scalar

    evals = scalar * probs + (1 - scalar) * rewards
    return evals

def getEvalsEmpirical(moves, net, game, p):
    #   Compute a probability distribution across legal moves
    outputs = net(game.toNN_vecs(every=False)[0], training=False)[:2]
    probs = policy_net.AdjustPolicy(outputs, moves, game.board)

    #   Get "empirical" rewards resulting from each move
    rewards = np.array(
        [game.getReward(m, p['mateReward'], simple=True)[0] for m in moves]
    )
    if not game.whiteToMove:
        rewards *= -1

    VAR_RATIO = 2.414 # 2.727 'tf_ex_compare'; 2.418 'tf_ex_gen_deep'

    evals = net.policy_certainty * probs + \
            VAR_RATIO * (1 - net.policy_certainty) * rewards
    return evals

#   For debugging/ testing purposes, return evals based on the move names in
#   alphabetical order (e.g. 'Re2' receives a worse eval than 'Bf5')
def getEvalsDebug(moves, net, game, p):
    move_names = [m.getMoveName(game) for m in moves]

    evals = np.linspace(0, 1, len(move_names))[
        [misc.match(x, sorted(move_names, reverse = True)) for x in move_names]
    ]
    return evals
