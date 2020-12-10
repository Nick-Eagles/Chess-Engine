import sys
sys.path.append('./')

import Traversal
import Game
import board_helper
import input_handling
import Session
import Move
import file_IO
import misc

#   Test Traversal.traverse(); in particular, for a Traversal 'trav':
#       1. trav.baseR holds the correct value

net_name = 'tf_profile'
tol = 0.00001

################################################################################
#   Test trav.baseR: position 1
################################################################################

#   Manually produce a particular position
game = Game.Game()
game.board = [[0] * 8,
              [0] * 8,
              [0] * 6 + [3, 4],
              [0] * 5 + [1, 0, 0],
              [0] * 8,
              [0] * 8,
              [0, 1, -1, -1, -4, 0, 0, 0],
              [6, -1, 0, -6] + [0] * 4]
game.invBoard = board_helper.invert(game.board)
game.wPieces = [2, 0, 0, 1, 1, 0]
game.bPieces = [3, 0, 0, 0, 0, 1]
game.canW_K_Castle = False
game.canW_Q_Castle = False
game.canB_K_Castle = False
game.canB_Q_Castle = False
game.updateValues()

#   Load configuration and fix important values
p = input_handling.readConfig(3)
p.update(input_handling.readConfig(1))
p['depth'] = 3
p['breadth'] = 15
p['gamma_exec'] = 0.9
p['mateReward'] = 3
p['minCertainty'] = 0.02

session = Session.Session([], [])
session.Load('nets/' + net_name, lazy=True)

#   Manually perform the optimal sequence from the starting position
print("Testing 'baseR' attribute of a traversal object (1)...")
expected_r1 = 0
g = game.copy()
file_IO.toFEN(g.toNN_vecs(every=False)[0], 'visualization/misc_tests/traverse_unit_tests.fen')

m = Move.Move((2, 7), (7, 7), 4) # Rh8+
expected_r1 += g.getReward(m, p['mateReward'], simple=True)[0]
g.doMove(m)

m = Move.Move((6, 4), (7, 4), -4) # Rh5 (forced)
expected_r1 += p['gamma_exec'] * g.getReward(m, p['mateReward'], simple=True)[0]
g.doMove(m)

m = Move.Move((2, 6), (3, 7), 3) # Bd8#
expected_r1 += p['gamma_exec']**2 * g.getReward(m, p['mateReward'], simple=True)[0]
g.doMove(m)

assert g.gameResult == 1, g.gameResult

#   Compute what the reward should be for the associated traversal
expected_r = p['mateReward'] * p['gamma_exec']**(p['depth'] - 1)
assert abs(expected_r1 - expected_r) < tol, abs(expected_r1 - expected_r)

trav = Traversal.Traversal(game, session.net, p)
trav.traverse()

misc.expect_equal(expected_r, trav.baseR, is_float=True)

################################################################################
#   Test trav.baseR: position 2
################################################################################

#   Manually produce a particular position
game = Game.Game()
game.board = [[6] + [0] * 7,
              [0] * 8,
              [0] * 8,
              [0] * 8,
              [0] * 8,
              [0] * 8,
              [0] * 5 + [-6, 0, 0],
              [0] * 7 + [-2]]
game.invBoard = board_helper.invert(game.board)
game.wPieces = [0, 0, 0, 0, 0, 0]
game.bPieces = [0, 1, 0, 0, 0, 0]
game.canW_K_Castle = False
game.canW_Q_Castle = False
game.canB_K_Castle = False
game.canB_Q_Castle = False
game.updateValues()

#   Do not incorporate NN evaluations in traversals
temp = session.net.certainty
session.net.certainty = 0

trav = Traversal.Traversal(game, session.net, p)
trav.traverse()

session.net.certainty = temp

print("Testing 'baseR' attribute of a traversal object (2)...")
misc.expect_equal(0, trav.baseR, is_float=True)

################################################################################
#   Test Traversal.processNode
################################################################################
