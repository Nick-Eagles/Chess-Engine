import sys
sys.path.append('./')
sys.path.append('./experimental')
import numpy as np
import tensorflow as tf

import read_pgn
import Game
import misc
import policy_net
import input_handling
import Session
import board_helper

GAME_PATH = 'external/2019_games_processed_t.txt'
NUM_GAMES = 1400
NET_PATH = 'nets/tf_ex_compare'

def process_line(game_str, obs_var, prob_var, p, net):
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
    #   Perform the game with the specified moves
    ############################################################################
    game = Game.Game()

    in_vecs = []
    moves_list = []
    
    for move_name in move_names:
        #   Verify that the move played was legal
        moves = board_helper.getLegalMoves(game)
        actual_names = [m.getMoveName(game.board) for m in moves]
        if move_name not in actual_names:
            move_name = read_pgn.fix_move_name(move_name, actual_names, p, game)

        #   Defer computation of variance across move probabilities until the
        #   end of the game
        moves_list.append(moves)
        in_vecs.append(game.toNN_vecs(every=False)[0])

        #   Compute variance across "empirical" rewards resulting from each move
        rewards = np.array([game.getReward(m, p['mateReward'], simple=True)[0]
                            for m in moves])
        obs_var.append(np.var(rewards))
        
        #   Do the move specified in the PGN
        move_played = moves[misc.match(move_name, actual_names)]
        game.doMove(move_played)

    #   Compute variance of probability distribution across legal moves (for
    #   every position in the last game)
    in_vecs = tf.stack([tf.reshape(x, [839]) for x in in_vecs])
    outputs = net(in_vecs, training=False)[:3]
    for i in range(outputs[0].shape[0]):
        these_outs = [tf.reshape(x[i, :], [1, -1]) for x in outputs]
        probs = policy_net.AdjustPolicy(these_outs, moves_list[i])
        prob_var.append(np.var(probs))
        

#   Load the entire file of PGN data (in text form) into memory
print("Reading in PGN...")
with open(GAME_PATH, 'r') as pgn_file:
    lines = pgn_file.readlines()[:NUM_GAMES]

print('Loading model...')
session = Session.Session([], [])
session.Load(NET_PATH)
p = input_handling.readConfig()

#   Form a list of variances for rewards received from performing each legal
#   move in a position, and a list of variances across the policy vector for
#   each position, respectively
print("Computing variances...")
obs_var = []
prob_var = []

for i, game_str in enumerate(lines):
    if i%100 == 0:
        print('Processing game ', i, '...', sep='')
        
    process_line(game_str, obs_var, prob_var, p, session.net)

assert len(obs_var) == len(prob_var)
print("Considered", len(obs_var), "positions.")

print("Computing means...")
print("Across move rewards:", np.mean(np.array(obs_var)))
print("Across probabilities:", np.mean(np.array(prob_var)))
