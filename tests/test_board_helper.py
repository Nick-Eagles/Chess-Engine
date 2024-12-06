import sys
sys.path.append('.')

import board_helper
import Move
import Game

################################################################################
#   Test 'inCheck'
################################################################################

def test_inCheck():
    #   Starting position
    board = [
        [4, 1, 0, 0, 0, 0, -1, -4],
        [2, 1, 0, 0, 0, 0, -1, -2],
        [3, 1, 0, 0, 0, 0, -1, -3],
        [5, 1, 0, 0, 0, 0, -1, -5],
        [6, 1, 0, 0, 0, 0, -1, -6],
        [3, 1, 0, 0, 0, 0, -1, -3],
        [2, 1, 0, 0, 0, 0, -1, -2],
        [4, 1, 0, 0, 0, 0, -1, -4]
    ]
    err_str = "Found white to be in check in the starting position"
    assert not board_helper.inCheck(board), err_str

    #   Somewhat complex position that isn't check
    board = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -6, 0, -5, 0],
        [0, 0, 0, 0, -4, 0, 0, 0],
        [0, 0, 0, -2, 0, 6, -1, 0],
        [0, 0, 0, 0, -2, -3, -4, 0]
    ]
    err_str = "Found white to be in check incorrectly"
    assert not board_helper.inCheck(board), err_str


    #   Check by pawn
    board = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0],
        [0, -6, 0, 0, 0, 6, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    err_str = "Failed to see a check by pawn"
    assert board_helper.inCheck(board), err_str

    #   Check by knight
    board = [
        [0, 0, 0, 6, 0, -6, 0, 0],
        [0, 0, 0, 0, 0, -2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    err_str = "Failed to see a check by knight"
    assert board_helper.inCheck(board), err_str

    #   Check by bishop
    board = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -6, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 6, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    err_str = "Failed to see a check by bishop"
    assert board_helper.inCheck(board), err_str

    #   Check by rook
    board = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -6, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 0, 0, 0, 0, -4]
    ]
    err_str = "Failed to see a check by rook"
    assert board_helper.inCheck(board), err_str

################################################################################
#   Test 'canCastle'
################################################################################

def test_canCastle():
    #   First, move white and black pawns forward on H file, then fianchetto
    #   both bishops, and finally move both knights. At that point both sides
    #   can kingside castle
    g = Game.Game(quiet=True)
    g.doMove(Move.Move((7, 1), (7, 2), 1))  # h3
    g.doMove(Move.Move((7, 6), (7, 4), -1)) # h5
    g.doMove(Move.Move((6, 1), (6, 2), 1))  # g3
    g.doMove(Move.Move((6, 6), (6, 5), -1)) # g6
    g.doMove(Move.Move((5, 0), (6, 1), 3))  # Bg2
    g.doMove(Move.Move((5, 7), (6, 6), -3)) # Bg7
    g.doMove(Move.Move((6, 0), (5, 2), 2))  # Nf3
    g.doMove(Move.Move((6, 7), (5, 5), -2)) # Nf6
    assert board_helper.canCastle(g) == (True, False), board_helper.canCastle(g)
    g.doMove(Move.Move((4, 1), (4, 3), 1))  # e4
    assert board_helper.canCastle(g) == (True, False), board_helper.canCastle(g)

    #   On a copy of the above game, move both kingside rooks, after which point
    #   neither side can castle. Have black do a dummy move afterward so that
    #   it's white's turn
    g2 = g.copy()
    g2.doMove(Move.Move((7, 7), (7, 6), -4)) # Rh7
    g2.doMove(Move.Move((7, 0), (7, 1), 4))  # Rh2
    assert board_helper.canCastle(g2) == (False, False), board_helper.canCastle(g)
    g2.doMove(Move.Move((1, 7), (2, 5), -2)) # Nc6
    assert board_helper.canCastle(g2) == (False, False), board_helper.canCastle(g)

    #   Even if rooks move back to their starting squares, neither side can
    #   castle. Again perform a dummy move so it's black's turn
    g2.doMove(Move.Move((7, 1), (7, 0), 4)) # Rh1
    g2.doMove(Move.Move((7, 6), (7, 7), -4)) # Rh8
    assert board_helper.canCastle(g2) == (False, False), board_helper.canCastle(g)
    g2.doMove(Move.Move((1, 0), (2, 2), 2)) # Nc3
    assert board_helper.canCastle(g2) == (False, False), board_helper.canCastle(g)

    #   Now create an artificial position where white can only not castle
    #   because the king would move through check
    g = Game.Game(quiet=True)
    g.board = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 0, 0, -6],
        [0, 0, -5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0]
    ]
    assert board_helper.canCastle(g2) == (False, False), board_helper.canCastle(g)
