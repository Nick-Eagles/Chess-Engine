import numpy as np
import Move
import Game

g = Game.Game()
m = Move.Move((0, 1), (0, 3), 1)
g2 = g.copy()
g2.doMove(m)
print(g.lastMove.toString())
print("So copied game's lastMove obj doesn't update original game's.")

g = Game.Game()
m = Move.Move((0, 1), (0, 3), 1)
g2 = g.copy()
g.doMove(m)
print(g2.lastMove.toString())
