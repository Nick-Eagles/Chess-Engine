import numpy as np
from scipy.special import expit
import tensorflow as tf

#   Given a Move to perform and a raw (cumulative) reward observed given that
#   move and future moves (before calling expit/sigmoid), return the associated
#   "label"
def ToOutputVec(move, r):
    out_vecs = [
        np.zeros(4096),
        np.zeros(6),
        tf.constant(expit(r), shape=[1, 1], dtype=tf.float32)
    ]

    ############################################################################
    #   Encode the policy vector
    ############################################################################
    
    #   One-hot-encoded combined start and end squares
    index = 512 * move.startSq[0] + 64 * move.startSq[1] + 8 * move.endSq[0] + \
        move.endSq[1]
    out_vecs[0][index] = 1
    out_vecs[0] = tf.constant(out_vecs[0], shape=[1, 4096], dtype=tf.float32)

    #   One-hot-encoded end-piece vector
    out_vecs[1][abs(move.endPiece) - 1] = 1
    out_vecs[1] = tf.constant(out_vecs[1], shape=[1, 6], dtype=tf.float32)

    return out_vecs

#   outputs: list of policy-related NN outputs (length 2)
#   Returns a numpy array of probabilities alongside the passed legal moves list
def AdjustPolicy(outputs, legal_moves, board):
    move_squares, end_piece = outputs
    
    assert move_squares.shape == (1, 4096), move_squares.shape
    assert end_piece.shape == (1, 6), end_piece.shape

    #   Grab the raw probabilities for the legal moves
    new_policy = np.zeros(len(legal_moves))
    for i, m in enumerate(legal_moves):
        move_index = 512 * m.startSq[0] + 64 * m.startSq[1] + 8 * m.endSq[0] + \
            m.endSq[1]
        new_policy[i] = move_squares[0, move_index]

        #   For pawn promotion, adjust the probability of promoting to this
        #   specific piece
        if abs(board[m.startSq[0]][m.endSq[1]]) != abs(m.endPiece):
            new_policy[i] *= end_piece[0, abs(m.endPiece) - 1]
        

    #   Normalize, since probabilities for illegal combinations will have a
    #   nonzero sum (though it should become smaller with training)
    new_policy /= np.sum(new_policy)

    return new_policy