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

sys.path.append('..')

import Game
import board_helper
import misc
import policy_net
import Move

pgn_path = here('external', '6980_games.txt')

def game_to_pairs(game_str):    
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
        in_vecs.append(game.toNN_vecs())
        
        #   Verify that the move played was legal
        moves = board_helper.getLegalMoves(game)
        actual_names = [m.getMoveName(game) for m in moves]
        assert move_name in actual_names, f'{move_name}; {" ".join(actual_names)}'
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
for game in games:
    temp = game_to_pairs(game)
    in_vecs += temp[0]
    out_vecs += temp[1]
    