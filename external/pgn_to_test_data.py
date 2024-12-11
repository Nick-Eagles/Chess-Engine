#   The goal here is to read in 6980_games.txt and produce tensors for training
#   a neural network. The output tensors are defined in an intentionally simple
#   and artificial way: if the current color has a queen, the policy should be
#   moving it to A1, and the evaluation should be that the current color wins.
#   Otherwise, move the current king to B2 and the opposite color wins. The idea
#   is that training a network to find the optimal chess move is highly
#   difficult, and an easier case should be attempted to ensure the foundation
#   is programmed correctly (the tensorflow piece and game.to_NNvecs())

from pyhere import here
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import gzip

sys.path.append(str(here()))

import Game
import board_helper
import misc
import policy_net
import Move

pgn_path = here('external', '6956_games.txt')
test_size = 0.1
random_state = 0
out_path = here('external', 'tensor_list.pkl.gz')

def game_to_pairs(game_str, j):    
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

with open(pgn_path, 'r') as f:
    games = f.read().splitlines()

in_vecs = []
out_vecs = []
for i, game in enumerate(games):
    temp = game_to_pairs(game, i)
    in_vecs.append(temp[0])
    out_vecs.append(temp[1])
    if i % 100 == 0:
        print(f'Done processing game {i}')

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

#   Write the tensors to disk
data = (X_train, X_test, y_train, y_test)
with gzip.open(out_path, 'wb') as f:
    pickle.dump(data, f)
