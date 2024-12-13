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

#   Given a single game as a string, return input and output tensors for
#   training a policy-value network. Output tensors are "synthetic" in that
#   rather than representing good moves and expected rewards, they are
#   intentionally simple-- see pgn_to_synthetic_data.py for context.
#
#   game_str: a string containing an entire game. See preprocess_pgn.R for
#       conventions on what is expected here
#   j: (int) game number for reporting back where illegitimate moves occur if
#       they are encountered
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

#   Given a single game as a string, return input and output tensors for
#   training a policy-value network.
#
#   game_str: a string containing an entire game. See preprocess_pgn.R for
#       conventions on what is expected here
#   p: a hyperparameter dictionary read in from config.txt
#   j: (int) game number for reporting back where illegitimate moves occur if
#       they are encountered
def game_to_pairs_real(game_str, p, j):
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
    #   Walk through the game to get a series of games, moves, and rewards
    ############################################################################

    game = Game.Game()
    g_list = []
    move_list = []
    r_list = []
    for i, move_name in enumerate(move_names):
        #   Verify that the move played was legal
        moves = board_helper.getLegalMoves(game)
        actual_names = [m.getMoveName(game) for m in moves]
        assert move_name in actual_names, f'game {j}: {move_name}; {" ".join(actual_names)}'
        move_played = moves[misc.match(move_name, actual_names)]

        move_list.append(move_played)
        g_list.append(game.copy())

        #   Do move and receive reward
        r = game.getReward(
            move_played, p['mateReward'], simple=True, copy=False
        )[0]
        r_list.append(r)

    ############################################################################
    #   Update and unravel reward sequence
    ############################################################################

    #   Update final reward, considering resignation as loss and agreed draws
    #   as legitimate draws
    if abs(stated_result) == 1:
        r_list[-1] = stated_result * p['mateReward']
    else:
        r_list[-1] = float(np.log(game.bValue / game.wValue))
    
    #   Unravel reward, passing it back from the gane's ending with a decay
    #   factor of gamma
    for i in range(1, len(r_list)):
        r_list[-1-i] += p['gamma'] * r_list[-i]

    ############################################################################
    #   Walk back through game to generate input and output tensors
    ############################################################################

    in_vecs = []
    out_vecs = []
    for i in range(len(r_list)):
        in_vecs.append(g_list[i].toNN_vecs(every = False)[0])
        out_vecs.append(
            policy_net.ToOutputVec(g_list[i], move_list[i], r_list[i])
        )

    return (in_vecs, out_vecs)

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
