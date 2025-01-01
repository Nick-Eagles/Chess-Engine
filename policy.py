import numpy as np
import tensorflow as tf

import policy_net
import misc
import board_helper

################################################################################
#   Functions for sampling several legal moves (subsetting search tree)
################################################################################

#   All functions have (net, game, p) as inputs, and output a list of Moves

#   Return the top p['breadth'] moves by evaluation
def sampleMovesStatic(net, game, p):
    moves = board_helper.getLegalMoves(game)

    #   Simply choose all moves if the breadth spans this far
    if p['breadth'] >= len(moves):
        return moves

    #   Otherwise return the set of highest-evaluated moves
    evals = p['evalFun'](moves, net, game, p)
    return [moves[i] for i in misc.topN(evals, p['breadth'])]

################################################################################
#   Functions for picking a single legal move to play
################################################################################

#   All functions have (net, game, p) as inputs, and output a Move

def getBestMoveRawPolicy(net, game, p):
    moves = board_helper.getLegalMoves(game)
    
    #   Compute a probability distribution across legal moves
    outputs = net(game.toNN_vecs(), training=False)[:2]
    probs = policy_net.AdjustPolicy(outputs, moves, game)

    return moves[np.argmax(probs)]

################################################################################
#   Functions for evaluating all legal moves
################################################################################

#   All functions have (moves, net, game, p) as inputs and return a numpy
#   array (same length as number of moves) of evaluations. Best evaluations are
#   always larger, regardless of the player to move

#   Simply try each move and check the network's value at the resulting
#   positions
def getEvalsValue(moves, net, game, p):
    #   Compute NN evaluations on each move
    r_real = np.zeros(len(moves))
    net_inputs = []
    for i, m in enumerate(moves):
        r, vec = game.getReward(m, p['mateReward'])
        r_real[i] = r
        net_inputs.append(tf.reshape(vec, (774,)))

    #   Get the value output from the network
    coef = 2 * game.whiteToMove - 1
    value = net(tf.stack(net_inputs), training=False)[-1]
    evals = r_real + p['gamma_exec'] * coef * value
    
    if not game.whiteToMove:
        evals = -1 * evals

    assert evals.shape == (len(moves),), evals.shape
    return evals

#   Evaluations directly use the policy piece of the network's output
def getEvalsPolicy(moves, net, game, p):
    #   Compute a probability distribution across legal moves
    outputs = net(game.toNN_vecs(), training=False)[:2]
    evals = policy_net.AdjustPolicy(outputs, moves, game)
    
    return evals

#   Evaluations weight the network policy against the observed rewards for
#   performing each move. The policy piece dominates when the best observed
#   reward is low, but observed rewards dominate when the largest is large
def getEvalsHybrid(moves, net, game, p):
    #   Compute a probability distribution across legal moves
    outputs = net(game.toNN_vecs(), training=False)[:2]
    probs = policy_net.AdjustPolicy(outputs, moves, game)

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

    evals = scalar * probs * p['mateReward'] + (1 - scalar) * rewards
    return evals

#   For debugging/ testing purposes, return evals based on the move names in
#   alphabetical order (e.g. 'Re2' receives a worse eval than 'Bf5')
def getEvalsDebug(moves, net, game, p):
    move_names = [m.getMoveName(game) for m in moves]

    evals = np.linspace(0, 1, len(move_names))[
        [misc.match(x, sorted(move_names, reverse = True)) for x in move_names]
    ]
    return evals
