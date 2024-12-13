import sys
sys.path.append('./')
sys.path.append('./experimental')
import numpy as np

import board_helper
import Game
import misc
import policy_net

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

        r = game.getReward(
            move_played, p['mateReward'], simple=True, copy=False
        )[0]
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

