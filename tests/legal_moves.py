import sys
import numpy as np

sys.path.append('../')
import board_helper
import board_helper_old
import Game


net_name = 'res_profile'

def is_in(move, move_set):
    return any([move.equals(x) for x in move_set])

def is_equal_set(m1, m2):
    m1_sub_m2 = all([is_in(x, m2) for x in m1])
    m2_sub_m1 = all([is_in(x, m1) for x in m2])
    return m1_sub_m2 and m2_sub_m1

game = Game.Game(quiet=False)

while (game.gameResult == 17):
    m1 = board_helper.getLegalMoves(game)
    m2 = board_helper_old.getLegalMoves(game)
    if not is_equal_set(m1, m2):
        m1Names = [x.getMoveName(game.board) for x in m1]
        print("\nm1 is:\n", ', '.join(m1Names))

        m2Names = [x.getMoveName(game.board) for x in m2]
        print("\nm2 is:\n", ', '.join(m2Names))
        game.toPGN('../visualization/latest_game.pgn')
        exit(1)
    #assert is_equal_set(m1, m2), "m1 is" + m1. + "m2 is" + m2

    game.doMove(m1[np.random.randint(len(m1))])

print(game.annotation)
