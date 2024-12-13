import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pyhere import here

sys.path.append(str(here()))
import board_helper
import Game
import misc
import policy_net
import Move

#   See 'pgn_to_test_data.py' for an explanation of this function
def game_to_pairs_synthetic(game_str, j):
    move_names = game_str.split(' ')

    stated_result = move_names.pop()
    if stated_result == '1-0':
        stated_result = 1
    elif stated_result == '0-1':
        stated_result = -1
    else:
        assert stated_result == '1/2-1/2', '"' + stated_result + '"'
        stated_result = 0

    ############################################################################
    #   Perform the game with the specified moves, generating examples at
    #   each move
    ############################################################################

    game = Game.Game()
    in_vecs = []
    out_vecs = []
    for i, move_name in enumerate(move_names):
        in_vecs.append(game.toNN_vecs(every = False)[0])
        
        #   Verify that the move played was legal
        moves = board_helper.getLegalMoves(game)
        actual_names = [m.getMoveName(game) for m in moves]
        assert move_name in actual_names, f'game {j}: {move_name}; {" ".join(actual_names)}'
        move_played = moves[misc.match(move_name, actual_names)]

        #   If the current color has a queen, create synthetic output that has
        #   move Qa1 and reward expit(100), else use move Kb2 with reward
        #   expit(-100). Negate reward for black
        if game.whiteToMove:
            if game.wPieces[5] > 0:
                start_sq = tuple(
                    [int(x[0]) for x in np.where(np.array(game.board) == 5)]
                )

                fake_move = Move.Move(start_sq, (0, 0), 5, validate = False)
                r = 100
            else:
                start_sq = tuple(
                    [int(x[0]) for x in np.where(np.array(game.board) == 6)]
                )
                fake_move = Move.Move(start_sq, (1, 1), 6, validate = False)
                r = -100
        else:
            if game.bPieces[5] > 0:
                start_sq = tuple(
                    [int(x[0]) for x in np.where(np.array(game.board) == -5)]
                )

                fake_move = Move.Move(start_sq, (0, 0), -5, validate = False)
                r = -100
            else:
                start_sq = tuple(
                    [int(x[0]) for x in np.where(np.array(game.board) == -6)]
                )
                fake_move = Move.Move(start_sq, (1, 1), -6, validate = False)
                r = 100

        out_vecs.append(policy_net.ToOutputVec(game, fake_move, r))
        
        game.doMove(move_played)
    
    return (in_vecs, out_vecs)

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

#   Convert many games, as output by generate_raw_pairs or similar, to a tuple
#   of tensors (X_train, X_test, y_train, y_test).
#
#   in_vecs: list of games (list) of positions (tensors) to use an inputs to
#       a policy-value neural network
#   out_vecs: a list of games (list) of output-layer (lists) of tensors to use
#       as outputs to a policy-value neural network
#   test_size: float passed to sklearn.model_selection.train_test_split
#   random_state: int passed to sklearn.model_selection.train_test_split
def games_to_tensors(in_vecs, out_vecs, test_size, random_state):
    #   Split data by game, not position, at first
    X_train, X_test, y_train, y_test = train_test_split(
        in_vecs, out_vecs, test_size = test_size, random_state = random_state
    )

    #   Format as (N, 839) tensors (includes N positions). The second index (first
    #   axis) of each tensor is the second position in the first game
    X_train = tf.stack(
        [tf.reshape(pos, (839)) for game in X_train for pos in game], axis = 0
    )
    X_test = tf.stack(
        [tf.reshape(pos, (839)) for game in X_test for pos in game], axis = 0
    )

    #   Format as lists of (N, Mi) tensors (for N positions for different values
    #   of Mi for each output component)
    y_train = [
        tf.stack(
            [
                tf.reshape(game[i][0], (64))
                for game in y_train for i in range(len(game))
            ]
        ),
        tf.stack(
            [
                tf.reshape(game[i][1], (64))
                for game in y_train for i in range(len(game))
            ]
        ),
        tf.stack(
            [
                tf.reshape(game[i][2], (6))
                for game in y_train for i in range(len(game))
            ]
        ),
        tf.stack(
            [
                tf.reshape(game[i][3], (1))
                for game in y_train for i in range(len(game))
            ]
        )
    ]

    y_test = [
        tf.stack(
            [
                tf.reshape(game[i][0], (64))
                for game in y_test for i in range(len(game))
            ]
        ),
        tf.stack(
            [
                tf.reshape(game[i][1], (64))
                for game in y_test for i in range(len(game))
            ]
        ),
        tf.stack(
            [
                tf.reshape(game[i][2], (6))
                for game in y_test for i in range(len(game))
            ]
        ),
        tf.stack(
            [
                tf.reshape(game[i][3], (1))
                for game in y_test for i in range(len(game))
            ]
        )
    ]

    return (X_train, X_test, y_train, y_test)
