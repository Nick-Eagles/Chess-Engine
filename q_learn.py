import Game
import buffer
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
        
    NN_vecs, rewards = produce_raw_pairs(net, p)

    #   For each game, propagate rewards backward with a decay factor of
    #   p['gamma']
    for i in range(len(rewards)):
        for j in range(1, len(rewards[i])):
            rewards[i][-1-j] += rewards[i][-j] * p['gamma']

    #   Flatten lists so they no longer are separated by game
    NN_vecs_out = []
    rewards_out = []
    for i in range(len(rewards)):
        rewards_out += rewards[i]
        NN_vecs_out += NN_vecs[i]

    assert len(rewards_out) == len(NN_vecs_out)

    data = [[],[],[]]
    for i, r in enumerate(rewards_out):
        num_examples = len(NN_vecs_out[i])

        #   Based on the number of examples for this particular position,
        #   assign the examples to the appropriate data buffer
        if num_examples == 2:
            buffer_index = 0
        elif num_examples == 4:
            buffer_index = 1
        elif num_examples == 16:
            buffer_index = 2
        else:
            sys.exit(
                "Received an invalid number of augmented positions" + \
                "associated with one example: " + str(num_examples)
            )

        #   Add the data to the correct buffer
        data[buffer_index] += [(NN_vecs_out[i][j], index_to_label(j, r))
                               for j in range(len(NN_vecs_out[i]))]

    buffer.verify(data, p, 3)
    
    return data

#   The first component of generateExamples, returning a tuple:
#   (list of lists of rewards received at each discrete move: one element of
#    the outer list corresponds to one game, list of lists of lists of NN
#    inputs: the outer two lists correspond to a particular game and position, 
#    respectively)
def produce_raw_pairs(net, p):
    game = Game.Game()
    step = 0

    #   Each element is a list representing rewards for moves during one game
    rewards = [[]]

    #   This follows 'rewards'
    NN_vecs = [[]]

    #   Continue taking actions and receiving rewards (start a new game if
    #   necessary)
    while step < p['maxSteps']:
        while (game.gameResult == 17 and step < p['maxSteps']):
            NN_vecs[-1].append(game.toNN_vecs())
            
            bestMove = policy.getBestMoveTreeEG(net, game, p)

            #   Do the move and append reward received
            r = game.getReward(
                bestMove, p['mateReward'], simple=True, copy=False
            )[0]
            rewards[-1].append(r)
            
            step += 1    

        #   Trim end of game if applicable
        if game.gameResult == 17:
            if len(rewards[-1]) <= p['rDepthMin']:
                #   Throw this game out, as it is too short
                rewards.pop()
                NN_vecs.pop()
            else:
                #   Discard the last p['rDepthMin'] NN_vecs for this game,
                #   and "unravel" ending rewards until both have the same length
                NN_vecs[-1] = NN_vecs[-1][:(-1 * p['rDepthMin'])]
                for i in range(p['rDepthMin']):
                    rewards[-1][-1] += p['gamma'] * rewards[-1].pop()


        #   Begin a new game if one ended but we still have more steps to take
        #   (Note we can't skip this even when p['maxSteps'] - step <=
        #   p['rDepthMin'], since conceivably a checkmate can occur in under
        #   p['rDepthMin'] halfmoves)
        if step < p['maxSteps']:
            game = Game.Game()
            rewards.append([])
            NN_vecs.append([])

    #   Ensure lengths of examples match lengths of labels
    assert len(NN_vecs) == len(rewards), \
           str(len(NN_vecs)) + ' ' + str(len(rewards))
    assert all(
        [len(rewards[i]) == len(NN_vecs[i]) for i in range(len(rewards))]
    ), "Unequal number of examples and labels generated"
    
    return (NN_vecs, rewards)

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
    p = input_handling.readConfig()

    print("Performing Q-learning (" + str(p['baseBreadth']) + " tasks)...")
    if p['mode'] >= 2:
        start_time = time.time()

    #   Run asynchronous data generation
    data = [[], [], []]
    for i in range(p['baseBreadth']):
        this_data = generateExamples(net, p)
        for j in range(3):
            data[j] += this_data[j]

    buffer.verify(data, p, 3)

    if p['mode'] >= 2:
        elapsed = round(time.time() - start_time, 2)
        print(
            "Done in", elapsed, "seconds. Generated " + \
            str(sum([len(x) for x in data])) + " training examples.\n"
        )
    else:
        print(
            "Done. Generated " + str(sum([len(x) for x in data])) + \
            " training examples.\n"
        )

    print("Determining certainty of network on the generated examples...")
    getCertainty(net, data, p)

    return data

#   "outputs" and "labels" are numpy arrays of shape (n,), where n is the number
#   of training examples present.
def valueCertainty(outputs, labels, p):
    out_norm = np.linalg.norm(outputs)

    if round(float(out_norm), 5) == 0:
        print(
            "Warning: norm of output values is too small; setting " + \
            "certainty to 0."
        )
        certainty = 0
    else:
        certainty = np.dot(outputs, labels) / \
                    (np.linalg.norm(labels) * out_norm)

    if p['mode'] >= 2:
        #   Print first 2 moments of expected and actual rewards
        print(
            "Mean and var of expected reward: ", round(np.mean(labels), 4),
            "; ", round(np.var(labels), 4), sep=""
        )
        print(
            "Mean and var of actual reward: ", round(np.mean(outputs), 4),
            "; ", round(np.var(outputs), 4), sep=""
        )

    assert certainty >= -1 and certainty <= 1, certainty
    return certainty


#   "outputs" is a list of Tensors of shape [n, 64], [n, 64], and [n, 6],
#   respectively; "labels" is of the same format
def policyCertainty(outputs, labels, p):
    #   Confirm shape of arguments
    num_examples = outputs[0].shape[0]
    assert outputs[0].shape == (num_examples, 64), outputs[0].shape
    assert outputs[1].shape == (num_examples, 64), outputs[1].shape
    assert outputs[2].shape == (num_examples, 6), outputs[2].shape
    assert labels[0].shape == (num_examples, 64), labels[0].shape
    assert labels[1].shape == (num_examples, 64), labels[1].shape
    assert labels[2].shape == (num_examples, 6), labels[2].shape
    
    #   For each example and output type (start square, etc), take the dot
    #   product of the output and label. Then compute the mean of this metric
    #   across examples, and finally across output type.
    c = tf.reduce_mean(tf.reduce_sum(outputs[0] * labels[0], axis=1))
    c += tf.reduce_mean(tf.reduce_sum(outputs[1] * labels[1], axis=1))
    c += tf.reduce_mean(tf.reduce_sum(outputs[2] * labels[2], axis=1))
    c = float(c/3)
    
    assert c >= -1 and c <= 1, c
    return c


def getCertainty(net, data, p, greedy=True):
    #   Inference
    inputs = tf.stack([tf.reshape(x[0], [839]) for x in data[0]], axis=0)
    outputs = net(inputs, training=False)

    ############################################################################
    #   Get certainty
    ############################################################################

    has_policy = isinstance(outputs, list)
    
    if has_policy:
        #   Get value-associated certainty
        labels = logit(
            tf.stack(
                [tf.reshape(x[1][-1], [1]) for x in data[0]],
                axis=0
            )
        ).flatten()

        value_certainty = valueCertainty(
            logit(outputs[-1]).flatten(), labels, p
        )
        if p['mode'] >= 1:
            print('"Value" certainty:', round(value_certainty, 5))

        #   Get policy-associated certainty
        labels = []
        labels.append(
            tf.stack(
                [tf.reshape(x[1][0], [64]) for x in data[0]], axis=0
            )
        )
        labels.append(
            tf.stack(
                [tf.reshape(x[1][1], [64]) for x in data[0]], axis=0
            )
        )
        labels.append(
            tf.stack(
                [tf.reshape(x[1][2], [6]) for x in data[0]], axis=0
            )
        )
        
        policy_certainty = policyCertainty(outputs[:3], labels, p)

        if p['mode'] >= 1:
            print('"Policy" certainty:', round(policy_certainty, 5))

        #   Compute a combined metric, based on weighted importance
        certainty = (1 - p['policyWeight']) * value_certainty + \
                    p['policyWeight'] * policy_certainty
        last_certainty = (1 - p['policyWeight']) * net.value_certainty + \
                         p['policyWeight'] * net.policy_certainty
    else:
        labels = logit(
            tf.stack(
                [tf.reshape(x[1], [1]) for x in data[0]], axis=0
            )
        ).flatten()
        value_certainty = valueCertainty(logit(outputs), labels, p)
        certainty = value_certainty
        last_certainty = net.value_certainty

    ############################################################################
    #   Scale certainty if policy used was not entirely greedy
    ############################################################################
    
    if greedy and p['epsGreedy'] < 0.5:
        #   0.5 is an arbitrarily established cutoff to prevent catastrophic
        #   amplification of 'noise' in estimating true certainty
        certainty /= 1 - p['epsGreedy']
        value_certainty /= 1 - p['epsGreedy']
        if has_policy:
            policy_certainty /= 1 - p['epsGreedy']

    ############################################################################
    #   Set model attributes (an exponentially weighted moving average)
    ############################################################################
    
    net.certaintyRate = net.certaintyRate * p['persist'] + \
                        (certainty - last_certainty) * (1 - p['persist'])
    net.value_certainty = net.value_certainty * p['persist'] + \
                          value_certainty * (1 - p['persist'])
    if has_policy:
        net.policy_certainty = net.policy_certainty * p['persist'] + \
                               policy_certainty * (1 - p['persist'])
    
    if p['mode'] >= 1:
        print(
            "Certainty of network on", len(data[0]), "examples:",
            round(certainty, 5)
        )
        print("Moving value certainty:", round(net.value_certainty, 5))
        if has_policy:
            print("Moving policy certainty:", round(net.policy_certainty, 5))
        print(
            "Moving rate of certainty change:", round(net.certaintyRate, 5),
            "\n"
        )
