import sys
import numpy as np

sys.path.append('../')
import board_helper
import Game

net_name = 'res_profile'

game = Game.Game(quiet=False)

while (game.gameResult == 17):
    m1 = board_helper.getLegalMoves(game)
    m2 = board_helper.anyLegalMoves(game)
    assert (len(m1) > 0) == m2
    if (len(m1) > 0) != m2:
        m1Names = [x.getMoveName(game.board) for x in m1]
        print("\nm1 is:\n", ', '.join(m1Names))

        print("anyLegalMoves returns", m2)
        game.toPGN('../visualization/latest_game.pgn')
        exit(1)

    game.doMove(m1[np.random.randint(len(m1))])

print(game.annotation)
