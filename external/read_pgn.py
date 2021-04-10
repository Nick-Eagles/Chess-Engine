import sys
sys.path.append('./')
sys.path.append('./experimental')
import numpy as np
from scipy.special import logit, expit
import tensorflow as tf
import random
import _pickle as pickle

import board_helper
import input_handling
import buffer as Buffer
import q_learn
import Game
import misc
import policy_net


#   Given a string containing one entire game, process this into data accepted
#   by the engine, appending to an ongoing buffer.
def process_line(buffer, game_str, p, augment, output_type):
    raw_pairs = generate_raw_pairs(game_str, p, output_type)
    update_buffer(buffer, raw_pairs, p, augment)


#   Modify a move name provided in the external data to match the corresponding
#   move name provided by the engine. This is required in 2 situations:
#       1. The engine's name is incorrect (incompatible with algebraic notation
#          rules): Move.getMoveName does not check legality of seemingly
#          alternate moves (e.g. another piece of the same type and end square
#          is actually pinned)
#       2. The engine uses optional notation (i.e. adds " e.p." for en passant
#          captures) which the external data does not use
#       3. The external data uses incorrect algebraic notation in rare
#          scenarios (3 queens are present: one on e3, d5, and d1. When naming
#          the move from d1 to f3, the external data calls it "Qd1f3", though
#          the proper disambiguation is "Q1f3"
#   "move_name" is the external data's name for a move (a string);
#   "engine_names" is a list of strings representing the legal move names
#       according to the engine.
def fix_move_name(move_name, engine_names, p, game):
    assert move_name not in engine_names, \
           "Attempted to fix a perfectly legitimate move name."

    ############################################################################
    #   One possible issue is the engine's use of 'e.p.' in en passant captures
    ############################################################################
    
    if move_name + ' e.p.' in engine_names:
        if p['mode'] >= 2:
            print('Adding " e.p." to a move name...')
        move_name += ' e.p.'

    ############################################################################
    #   Another issue is the engine providing rank or file clarification
    #   in move names when not required (such as when the other move
    #   would be illegal due to a discovered check). Try removing this
    ############################################################################
    
    temp_engine_names = engine_names.copy()
    for i in range(len(temp_engine_names)):
        if len(temp_engine_names[i]) >= 4:
            orig_name = temp_engine_names[i]
            temp_engine_names[i] = temp_engine_names[i][0] + \
                                   temp_engine_names[i][2:]
            if move_name == temp_engine_names[i]:
                if p['mode'] >= 2:
                    print("Changing move name", move_name, "to", \
                          orig_name + '.')
                move_name = orig_name
                            
            temp_engine_names[i] = orig_name

    ############################################################################
    #   Finally, some move names in the data provide a file and rank to
    #   disambiguate, when only one is required (and correct)
    ############################################################################
    
    cond = len(move_name) >= 5 and \
           move_name[1] in 'abcdefgh' and \
           move_name[2] in '12345678'
    
    #   Try removing file specification
    if cond and move_name[0] + move_name[2:] in engine_names:
        if p['mode'] >= 2:
            print("Changing move name", move_name, "to", \
                  move_name[0] + move_name[2:] + '.')
        move_name = move_name[0] + move_name[2:]
    #   Try removing rank specification
    elif cond and move_name[:2] + move_name[3:] in engine_names:
        if p['mode'] >= 2:
            print("Changing move name", move_name, "to", \
                  move_name[:2] + move_name[2:] + '.')
        move_name = move_name[:2] + move_name[3:]    

    #   Other mismatches should not occur
    assert move_name in engine_names, \
            'Move in PGN: ' + move_name + '\n' + \
            'Legal moves known: ' + ','.join(engine_names) + '\n' + \
            game.verbosePrint()

    return move_name


#   Given a string containing one entire game ('game_str'), the hyperparameter
#   dictionary 'p', and an 'output_type' of "value": return a 2-tuple where the
#   first element is a list of (lists of) NN inputs, and the second element is
#   a list of immediate rewards received for the moves performed. If
#   'output_type' is "policy_value", return a 3-tuple with the same first two
#   elements, and the additional list of labels for the third element.
def generate_raw_pairs(game_str, p, output_type='value'):
    #   Basic checks on input parameters
    assert output_type == 'value' or output_type == 'policy_value', output_type
    
    move_names = game_str.split(' ')
    
    stated_result = move_names.pop()[:-1]
    if stated_result == '1-0':
        stated_result = 1
    elif stated_result == '0-1':
        stated_result = -1
    else:
        assert stated_result == '1/2-1/2', '"' + stated_result + '"'
        stated_result = 0

    ############################################################################
    #   Perform the game with the specified moves and receive rewards
    ############################################################################
    game = Game.Game()
    nn_vecs = []
    rewards = []
    out_vecs = []
    for i, move_name in enumerate(move_names):
        nn_vecs.append(game.toNN_vecs())
        
        #   Verify that the move played was legal
        moves = board_helper.getLegalMoves(game)
        actual_names = [m.getMoveName(game.board) for m in moves]
        if move_name not in actual_names:
            move_name = fix_move_name(move_name, actual_names, p, game)

        move_played = moves[misc.match(move_name, actual_names)]
        if output_type == 'policy_value':
            g = game.copy()

        r = game.getReward(move_played,
                           p['mateReward'],
                           simple=True,
                           copy=False)[0]
        rewards.append(r)

        if output_type == 'policy_value':
            out_vecs.append(policy_net.ToOutputVec(g, move_played, r))

    ############################################################################
    #   Update final reward, considering resignation as loss and agreed draws
    #   as legitimate draws
    ############################################################################
    if game.gameResult != stated_result:
        if abs(stated_result) == 1:
            rewards[-1] = stated_result * p['mateReward']
        else:
            #   For simplicity, we are neglecting how a potential capture in the
            #   last move affects the drawing reward (this really should be the
            #   'bValue' and 'wValue' before performing the last move, to keep
            #   things precisely consistent, though it shouldn't matter much)
            rewards[-1] = float(np.log(game.bValue / game.wValue))

    #   Verify lengths of data generated
    assert len(rewards) == len(move_names), len(move_names)-len(rewards)
    assert len(nn_vecs) == len(move_names), len(move_names)-len(nn_vecs)

    if output_type == 'policy_value':
        return (nn_vecs, rewards, out_vecs)
    else:
        return (nn_vecs, rewards)


#   Given the raw pairs of data (2-tuples) for a game, update the ongoing
#   data buffer with the processed data for that game
def update_buffer(buffer, raw_pairs, p, augment):
    if len(raw_pairs) == 2:
        nn_vecs, rewards = raw_pairs
    else:
        nn_vecs, rewards, out_vecs = raw_pairs

    for i in range(1, len(rewards)):
        rewards[-1-i] += rewards[-i] * p['gamma']

    if augment:
        assert len(raw_pairs) == 2, \
               "Augmentation not yet supported for policy-value networks"
        
        for i, r in enumerate(rewards):
            num_examples = len(nn_vecs[i])

            #   Based on the number of examples for this particular position,
            #   assign the examples to the appropriate data buffer
            if num_examples == 2:
                buffer_index = 0
            elif num_examples == 4:
                buffer_index = 1
            elif num_examples == 16:
                buffer_index = 2
            else:
                sys.exit("Received an invalid number of augmented positions" + \
                         "associated with one example: " + str(num_examples))

            #   Add the data to the correct buffer
            buffer[buffer_index] += [(nn_vecs[i][j],
                                      q_learn.index_to_label(j, r))
                                     for j in range(len(nn_vecs[i]))]

        Buffer.verify(buffer, p, 3)
    else:
        for i in range(len(rewards)):
            if len(raw_pairs) == 2:
                buffer[0].append((nn_vecs[i][0],
                                  tf.constant(expit(rewards[i]),
                                              shape=[1,1],
                                              dtype=tf.float32)))
            else:
                #   Use the cumulative reward propagated backward with gamma,
                #   not the immediate reward contained in out_vecs[i][3]
                this_r = tf.constant(expit(rewards[i]),
                                     shape=[1, 1],
                                     dtype=tf.float32)
                buffer[0].append((nn_vecs[i][0],
                                  out_vecs[i][:3] + [this_r]))

        Buffer.verify(buffer, p, 1)


#   Given a filename including preprocessed PGN data (space-delimited sequences
#   of move names followed by the game result), load this information into a
#   data buffer accepted by the engine.
def load_games(filename, p, line_nums, net, augment=False, certainty=True):
    #   Initialize the buffer
    if augment:
        buffer = [[], [], []]
    else:
        buffer = [[]]

    if 'policy_start_square' in [x.name for x in net.layers]:
        output_type = 'policy_value'
    else:
        output_type = 'value'

    #   Load the entire file of PGN data (in text form) into memory
    with open(filename, 'r') as pgn_file:
        lines = pgn_file.readlines()

    #   Add the data contained in each line individually
    for i in line_nums:
        process_line(buffer, lines[i], p, augment, output_type)

    if certainty:
        q_learn.getCertainty(net, buffer, p, greedy=False)
        
    return buffer


class RawDataGen(tf.keras.utils.Sequence):
    def __init__(self, batch_size, base_dir, n, shuffle = True):
        self.batch_size = batch_size
        self.base_dir = base_dir
        self.shuffle = shuffle
        self.n = n

        self.indices = list(range(n))
        if shuffle:
            random.shuffle(self.indices)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indices)

    def __getitem__(self, index):
        buffer = [[]]
        first_index = index * self.batch_size
        for i in range(self.batch_size):
            filename = self.base_dir + '/' + \
                       str(self.indices[first_index + i]) + '.pkl'
            with open(filename, 'rb') as f:
                buffer[0].append(pickle.load(f))

        return Buffer.collapse(buffer)

    def __len__(self):
        return self.n // self.batch_size
