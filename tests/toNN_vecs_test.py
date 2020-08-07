import sys
sys.path.append('../')

import Game
import file_IO
import Move

#   Test Game.toNN_vecs: namely that data augmentation is performed under the
#   expected circumstances and correctly when performed. This script takes
#   several hand-made positions and produces .fen files for the positions,
#   and the .fen files produced after compressing what's returned from the
#   function.

##############################################################################
#   The start position: no augmentation (aside from the inverted by color and
#   rank position) should be produced
##############################################################################

game = Game.Game()
vecs = game.toNN_vecs()
assert len(vecs) == 2, len(vecs)
filename0 = '../visualization/toNN_vecs_test/start.fen'
filename1 = '../visualization/toNN_vecs_test/start_invert.fen'
file_IO.toFEN(file_IO.compressNNinput(vecs[0]), filename0)
file_IO.toFEN(file_IO.compressNNinput(vecs[1]), filename1)

##############################################################################
#   Near the start, but kings have moved: this should generate 4 positions
##############################################################################

game = Game.Game()
#   Move pawns
game.doMove(Move.Move((4, 1), (4, 3), 1))
game.doMove(Move.Move((5, 6), (5, 5), -1))
#   Move kings up one square from back ranks
game.doMove(Move.Move((4, 0), (4, 1), 6))
game.doMove(Move.Move((4, 7), (5, 6), -6))

vecs = game.toNN_vecs()
assert len(vecs) == 4, len(vecs)
fileBase = '../visualization/toNN_vecs_test/moved_kings'
for i in range(4):
    file_IO.toFEN(file_IO.compressNNinput(vecs[i]), fileBase + str(i) + '.fen')

##############################################################################
#   Endgame: king and rook vs. king- this should generate 16 positions
##############################################################################
    
game = Game.Game()
game.board = [[0]*8, [6, 0, 0, 0, 0, 0, 0, -6], [0]*8, [4, 0, 0, 0, 0, 0, 0, 0],
              [0]*8, [0]*8, [0]*8, [0]*8]
game.canW_K_Castle = False
game.canW_Q_Castle = False
game.canB_K_Castle = False
game.canB_Q_Castle = False
game.wPieces = [0, 0, 0, 0, 1, 0]
game.bPieces = [0, 0, 0, 0, 0, 0]

vecs = game.toNN_vecs()
assert len(vecs) == 16, len(vecs)
fileBase = '../visualization/toNN_vecs_test/endgame'
for i in range(16):
    file_IO.toFEN(file_IO.compressNNinput(vecs[i]), fileBase + str(i) + '.fen')

