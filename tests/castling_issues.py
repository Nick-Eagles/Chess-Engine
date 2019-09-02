import sys
import numpy as np
sys.path.append('../')

import Network
import Game
import Move

#   Move queenside rook and verify that queenside castling is no longer
#   allowed
g = Game.Game(quiet=True)
g.doMove(Move.Move((0, 1), (0, 2), 1))
g.doMove(Move.Move((0, 6), (0, 4), -1))
g2 = g.copy()
g.doMove(Move.Move((0, 0), (0, 1), 4))
assert not g.canW_Q_Castle
assert g2.canW_Q_Castle
g2.doMove(Move.Move((0, 0), (0, 1), 4))
assert not g2.canW_Q_Castle

g = Game.Game()
g.doMove(Move.Move((1, 1), (1, 2), 1))
g.doMove(Move.Move((6, 6), (6, 5), -1))
g.doMove(Move.Move((7, 1), (7, 3), 1))
g.doMove(Move.Move((5, 7), (6, 6), -3))
g.doMove(Move.Move((7, 3), (7, 4), 1))
g.doMove(Move.Move((6, 6), (0, 0), -3))
assert not g.canW_Q_Castle
g2 = g.copy()
assert not g2.canW_Q_Castle

print("Success!")
