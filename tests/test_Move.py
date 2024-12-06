import sys
sys.path.append('.')

import Move
import Game

################################################################################
#   Test 'getMoveName', which summarizes most of the difficult/ error-prone
#   functionality in the Move class
################################################################################

def test_getMoveName():
    g = Game.Game(quiet=True)
    g.board = [
        [0, 0, 0, 0, 0, 2, 0, 0],
        [0, 5, 0, 5, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0, 0, 0],
        [0, 0, 6, 2, 0, -4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -5],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, -6, 0]
    ]

    #   Two different kinds of disambiguation
    assert Move.Move((1, 1), (2, 2), 5).getMoveName(g.board) == 'Qb2xc3'
    assert Move.Move((3, 1), (2, 1), 5).getMoveName(g.board) == 'Qdc2'

    #   Promotion and pawn capture
    assert Move.Move((6, 6), (6, 7), 4).getMoveName(g.board) == 'g8=R'
    assert Move.Move((6, 6), (5, 7), 2).getMoveName(g.board) == 'gxf8=N'

    #   Seemingly ambiguous, but not, because of a pin
    assert Move.Move((4, 3), (2, 4), 2).getMoveName(g.board) == 'Nc5'
