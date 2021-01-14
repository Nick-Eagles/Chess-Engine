import misc

import tensorflow as tf

#   Utilities for working with "buffers", the main in-memory representation of
#   data used by the program. Note that functions for reading and writing to
#   disk are included in file_IO.py and not here.
#
#   Buffers are lists of N lists of training examples. In main.py, N=4, and
#   each list of the 4 lists have a different decay rate. The first list
#   contains examples coming in pairs (the original example is augmented once);
#   the second has examples in groups of 4; the third in groups of 16, and the
#   4th are checkmates read from file (not generated in q_learn.py). In
#   practice, the checkmates begin as a buffer with N=1, before being merged.
#
#   Within each list, examples are 2-tuples: (input, label) pairs. Both are
#   tf.Tensors of batch dimension 1.

#   Combine buffers and reformat for compatibility with Keras
def collapse(buff):
    bigBuff = []
    for i in range(len(buff)):
        bigBuff += buff[i]

    x = tf.stack([tf.reshape(z[0], [839]) for z in bigBuff])

    if isinstance(bigBuff[0][1], list):
        #   Labels are for policy-value networks
        y = [tf.stack([tf.reshape(z[1][0], [64]) for z in bigBuff]),
             tf.stack([tf.reshape(z[1][1], [64]) for z in bigBuff]),
             tf.stack([tf.reshape(z[1][2], [6]) for z in bigBuff]),
             tf.stack([tf.reshape(z[1][3], [1]) for z in bigBuff])]
    else:
        #   Labels for value-only networks
        y = tf.stack([tf.reshape(z[1], [1]) for z in bigBuff])

    return (x, y)


#   Function for dropping a fraction of the existing examples, without
#   causing overrepresentation of augmented data examples in the steady-state
def filter(tBuffer, vBuffer, p):
    md = p['memDecay']
    
    #   The user intends to drop the fraction p['memDecay'] of total examples
    #   across the first 3 buffers. In practice, we must use different decay
    #   fractions for each buffer so that positions that generate more training
    #   examples are not overrepresented. Compute the decay fraction for the
    #   first buffer, which we will call 'md'.

    #   The number we want to drop
    num_to_drop = md * sum([len(x) for x in tBuffer[:3]])

    #   The number we would drop naively (note this can even be greater than the
    #   total amount, which does not present problems)
    num_dropped = md * len(tBuffer[0]) + \
                  md * len(tBuffer[1]) * 2 + \
                  md * len(tBuffer[2]) * 8

    #   Adjust 'md' to achieve the correct number to drop
    md *= num_to_drop / num_dropped
    
    for j in range(4):
        #   The fraction of each sub-buffer to keep. These are different
        #   to ensure positions able to be reflected/ rotated are not
        #   overrepresented in training
        if j == 0:
            frac_keep = 1 - md
        elif j == 1:
            frac_keep = max(1 - 2 * md, 0)
        elif j == 2:
            frac_keep = max(1 - 8 * md, 0)
        else: # j = 3
            frac_keep = 1 - p['memDecay']

        if j < 3:
            #   Recycle old validation examples for use in training buffers
            #   (but do not put old validation checkmates in training buffer)
            temp = misc.divvy(vBuffer[j], frac_keep)
            vBuffer[j] = temp[0]
            tBuffer[j] = misc.divvy(tBuffer[j], frac_keep, both=False)[0] + \
                         temp[1]
        else:
            tBuffer[j] = misc.divvy(tBuffer[j], frac_keep, both=False)[0]
            vBuffer[j] = misc.divvy(vBuffer[j], frac_keep, both=False)[0]

    verify(tBuffer, p)
    verify(vBuffer, p)


#   Check that a buffer appears to be formatted correctly, and exit the program
#   upon noticing problems
def verify(data, p, numBuffs=4):
    #   All buffers exist
    assert len(data) == numBuffs, len(data)
    for i in range(numBuffs):
        if len(data[i]) > 0:
            # the first example consists of an input and output
            assert len(data[i][0]) == 2, len(data[i][0])

            # the input is of proper shape
            assert data[i][0][0].shape == (1,839), data[i][0][0].shape

            #   Check label shape(s)
            if isinstance(data[i][0][1], list):
                #   The policy and value outputs are of proper shape
                assert data[i][0][1][0].shape == (1, 64), data[i][0][1][0].shape
                assert data[i][0][1][1].shape == (1, 64), data[i][0][1][1].shape
                assert data[i][0][1][2].shape == (1, 6), data[i][0][1][2].shape
                assert data[i][0][1][3].shape == (1, 1), data[i][0][1][3].shape
            else:
                #   The value output is of proper shape
                assert data[i][0][1].shape == (1, 1), data[i][0][1].shape
                
        elif p['mode'] >= 2:
            print("Warning: buffer", i, "was empty.")


#   Concatenate two buffers and return the result
def combine(buff1, buff2):
    assert len(buff1) == len(buff2), "Attempted to combine unequal buffer types"

    new_buff = []
    for i in range(len(buff1)):
        new_buff.append(buff1[i] + buff2[i])

    return new_buff
