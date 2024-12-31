from pyhere import here
import sys

sys.path.append(str(here()))
import Game
import Move
import policy_net

def tactics_to_pairs(fen_str, moves, p):
    in_vecs = []
    out_vecs = []

    #   The tactical position starts one move in. Create a Game at that point
    game = Game.Game.fromFEN(fen_str)
    game.doMove(Move.Move.from_uci(moves[0], game))
    in_vecs.append(game.toNN_vecs())

    #   Now gather total reward for performing the remaining tactical sequence
    game_copy = game.copy()
    gamma_coef = 1
    r = 0
    for move in moves[1:]:
        actual_move = Move.Move.from_uci(move, game_copy)
        r += gamma_coef * game_copy.getReward(
            actual_move, p['mateReward'], simple=True, copy=False
        )[0]
        gamma_coef *= p['gamma']

    #   Form output layer from first move and total reward over the sequence
    out_vecs.append(
        policy_net.ToOutputVec(Move.Move.from_uci(moves[1], game), r, game)
    )

    return (in_vecs, out_vecs)
