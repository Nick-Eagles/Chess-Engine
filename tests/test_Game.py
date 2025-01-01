import sys
from pyhere import here
sys.path.append(str(here()))

import Game

def test_fromFEN():
    #   Test a FEN of the starting position
    fen_str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    game1 = Game.Game()
    game2 = Game.Game.fromFEN(fen_str)

    assert game1.board == game2.board
    assert game1.invBoard == game2.invBoard
    assert game1.wPieces == game2.wPieces
    assert game1.bPieces == game2.bPieces
    assert game1.wValue == game2.wValue
    assert game1.bValue == game2.bValue

    assert game2.whiteToMove
    assert game2.canW_K_Castle
    assert game2.canW_Q_Castle
    assert game2.canB_K_Castle
    assert game2.canB_Q_Castle
    assert not game2.currentlyCheck
    assert game2.moveNum == 1
    assert game2.movesSinceAction == 0

def test_toFEN():
    #   Test starting position
    fen_str = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    actual_fen_str = Game.Game.fromFEN(fen_str).toFEN()
    assert actual_fen_str == fen_str, actual_fen_str

    #   Test slightly different position near the start position
    fen_str = 'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2'
    actual_fen_str = Game.Game.fromFEN(fen_str).toFEN()
    assert actual_fen_str == fen_str, actual_fen_str
