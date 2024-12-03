import numpy as np
from scipy.special import expit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import input_handling

def InitializeNet(p, output_type, numGroups=None, blocksPerGroup=None, blockWidth=None):
    if numGroups is None:
        #   Number of groups of residual blocks
        messDef = "Define network architecture: how many residual groups? "
        messOnErr = "Not a valid number."
        cond = 'var >= 0 and var < 50'
        numGroups = input_handling.getUserInput(messDef, messOnErr, 'int', cond)

    if blocksPerGroup is None:
        #   Number of blocks in a group
        messDef = "Number of residual blocks per group? "
        blocksPerGroup = input_handling.getUserInput(
            messDef, messOnErr, 'int', cond
        )

    if blockWidth is None:
        #   Layers per residual block
        messDef = "Number of layers in one residual block? "
        cond = 'var > 0 and var < 10'
        blockWidth = input_handling.getUserInput(
            messDef, messOnErr, 'int', cond
        )
        
    inputs = keras.Input(shape=(839,), name="game_position")
    x = inputs

    for i in range(numGroups):
        messDef = "Length of neurons in group " + str(i+1) + "? "
        cond = 'var > 0 and var < 10000'
        layLen = input_handling.getUserInput(
            messDef, "Not a valid number.", 'int', cond
        )

        #   Linear projection to match block input and output lengths
        x = layers.Dense(layLen, name='linear_projection' + str(i+1))(x)
        
        for j in range(blocksPerGroup):
            blockInput = x
            for k in range(blockWidth-1):
                x = layers.Dense(
                    layLen,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(p['weightDec'])
                )(x)
                x = layers.BatchNormalization(momentum=p['popPersist'])(x)
            x = layers.Dense(
                layLen,
                activation="relu",
                kernel_regularizer=regularizers.l2(p['weightDec'])
            )(x)

            #   Residual connection, with batch norm afterward
            layer_num = str(i*blocksPerGroup + j)
            x = layers.add([x, blockInput], name='residual_conn' + layer_num)
            x = layers.BatchNormalization(
                name='block_output' + layer_num,
                momentum=p['popPersist']
            )(x)

    if output_type == 'policy_value':
        #   Output layer (3 pieces of a policy vector and a scalar value)
        policy_start_sq = layers.Dense(
            64, activation="softmax", name="policy_start_square"
        )(x)
        policy_end_sq = layers.Dense(
            64, activation="softmax", name="policy_end_square"
        )(x)
        policy_end_piece = layers.Dense(
            6, activation="softmax", name="policy_end_piece"
        )(x)


        value = layers.Dense(
            1, activation="sigmoid", name="output",
            kernel_regularizer=regularizers.l2(p['weightDec'])
        )(x)

        net = keras.Model(
            inputs=inputs,
            outputs=[policy_start_sq, policy_end_sq, policy_end_piece, value],
            name="network"
        )

        net.policy_certainty = 0
    else:
        #   Output layer
        output = layers.Dense(
            1, activation="sigmoid", name="output",
            kernel_regularizer=regularizers.l2(p['weightDec'])
        )(x)

        net = keras.Model(inputs=inputs, outputs=output, name="network")
        
    net.value_certainty = 0
    net.certaintyRate = 0
    
    return net


#   output_type can be 'policy_value' or 'value'
def CompileNet(net, p, optim, output_type='value'):
    #   Basic checks on input parameters
    assert output_type == 'value' or output_type == 'policy_value', output_type

    if output_type == 'policy_value':
        w = p['policyWeight']
        loss_weights = [w/3, w/3, w/3, 1 - w]
    
    #   Compile model
    if output_type == 'value':
        loss = tf.keras.losses.BinaryCrossentropy()

        net.compile(
            optimizer = optim,
            loss = loss,
            metrics = [tf.keras.metrics.KLDivergence()]
        )
    else:
        loss = [
            tf.keras.losses.CategoricalCrossentropy(), # policy: start sq
            tf.keras.losses.CategoricalCrossentropy(), # policy: end sq
            tf.keras.losses.CategoricalCrossentropy(), # policy: end piece
            tf.keras.losses.BinaryCrossentropy()       # value
        ]

        net.compile(
            optimizer = optim,
            loss = loss,
            loss_weights = loss_weights,
            metrics = [tf.keras.metrics.CategoricalAccuracy()]
        )

#   Given a Game, a Move to perform from the position described in the Game, and
#   a raw (cumulative) reward observed given that move and future moves (before
#   calling expit/sigmoid), return the associated "label"
def ToOutputVec(game, move, r):
    out_vecs = [
        np.zeros(64),
        np.zeros(64),
        np.zeros(6),
        tf.constant(expit(r), shape=[1, 1], dtype=tf.float32)
    ]

    ############################################################################
    #   Encode the policy vector (the first 133 indices of 'out_vec')
    ############################################################################
    
    #   One-hot encoded start and end squares
    out_vecs[0][move.startSq[0] * 8 + move.startSq[1]] = 1
    out_vecs[0] = tf.constant(out_vecs[0], shape=[1, 64], dtype=tf.float32)
    out_vecs[1][move.endSq[0] * 8 + move.endSq[1]] = 1
    out_vecs[1] = tf.constant(out_vecs[1], shape=[1, 64], dtype=tf.float32)

    #   "Append" a one-hot encoded end piece vector (length 6, at indices
    #   [128, 131])
    out_vecs[2][abs(move.endPiece) - 1] = 1
    out_vecs[2] = tf.constant(out_vecs[2], shape=[1, 6], dtype=tf.float32)

    return out_vecs

#   outputs: list of policy-related NN outputs (length 3)
#   Returns a numpy array of probabilities alongside the passed legal moves list
def AdjustPolicy(outputs, legal_moves):
    start_square, end_square, end_piece = outputs
    
    assert start_square.shape == (1, 64), start_square.shape
    assert end_square.shape == (1, 64), end_square.shape
    assert end_piece.shape == (1, 6), end_piece.shape

    #   Take the outer product of all three policy vectors, so that the
    #   (i, j, k)th entry of the result is the probability of picking the
    #   ith start square, jth end square, and kth end piece
    raw_policy = (start_square.numpy().T \
                  @ end_square.numpy()).reshape(64, 64, 1) \
                  @ end_piece.numpy()

    assert raw_policy.shape == (64, 64, 6), raw_policy.shape
    temp = start_square[0, 7] * end_square[0, 13] * end_piece[0, 4]
    temp2 = abs(np.log(raw_policy[7, 13, 4] / temp))
    assert temp2 < 0.0001, temp2

    #   Grab the raw probabilities for the combinations of start+end square,
    #   end piece that are legal
    new_policy = np.zeros(len(legal_moves))
    for i, m in enumerate(legal_moves):
        new_policy[i] = raw_policy[
            m.startSq[0] * 8 + m.startSq[1],
            m.endSq[0] * 8 + m.endSq[1],
            abs(m.endPiece) - 1
        ]

    #   Normalize, since probabilities for illegal combinations will have a
    #   nonzero sum
    new_policy /= np.sum(new_policy)

    return new_policy
