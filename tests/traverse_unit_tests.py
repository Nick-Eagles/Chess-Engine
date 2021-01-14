import sys
sys.path.append('./')
import numpy as np
from scipy.special import logit

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
#   Test trav.baseR: position 1 (finding the optimal sequence)
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
p = input_handling.readConfig()
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
file_IO.toFEN(g.toNN_vecs(every=False)[0],
              'visualization/misc_tests/traverse_unit_tests1.fen')

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
#   Test trav.baseR: position 2 (finding the optimal sequence)
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
#   Test trav.baseR: position 3 (test inclusion of NN evals)
################################################################################

temp = session.net.certainty
session.net.certainty = 0.14

game = Game.Game()

p['depth'] = 1
p['breadth'] = 20
p['minCertainty'] = -0.1 # NN evals should be included

#   Manually find the expected base reward from the traversal by searching for
#   the top evaluation at all of the possible positions at depth 1
moves = board_helper.getLegalMoves(game)
nn_evals = np.zeros(len(moves))

assert len(moves) == 20, len(moves)

for i, m in enumerate(moves):
    g = game.copy()
    g.doMove(m)
    nn_evals[i] = session.net(g.toNN_vecs(every=False)[0], training=False)

nn_evals = session.net.certainty * p['gamma_exec'] * logit(nn_evals)
best_eval = float(np.max(nn_evals))
assert best_eval != 0, best_eval

#   Actually perform the traversal
trav = Traversal.Traversal(game, session.net, p)
trav.traverse()
session.net.certainty = temp

print("Testing 'baseR' attribute of a traversal object (3)...")
misc.expect_equal(best_eval, trav.baseR, is_float=True)

################################################################################
#   Test trav.baseR: position 4 (finding the optimal sequence; inclusion of
#   NN eval)
################################################################################

#   Manually produce a particular position
game = Game.Game()
game.board = [[0] * 8,
              [0] * 8,
              [0] * 6 + [3, 0],
              [0] * 8,
              [0] * 8,
              [4] + [0] * 7,
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

temp = session.net.certainty
session.net.certainty = 0.14

p['depth'] = 3
p['breadth'] = 21
p['gamma_exec'] = 0.9
p['mateReward'] = 3
p['minCertainty'] = 0.0

#   Visually show that it is as expected
file_IO.toFEN(game.toNN_vecs(every=False)[0],
              'visualization/misc_tests/traverse_unit_tests4.fen')

#   Perform the optimal sequence manually, then measure reward (including NN
#   eval)
g = game.copy()

num_moves = len(board_helper.getLegalMoves(g))
assert num_moves == 21, num_moves

m = Move.Move((2, 6), (3, 7), 3)
assert m.getMoveName(g.board) == 'Bd8'
g.doMove(m)

m = Move.Move((7, 3), (7, 4), -6)
assert m.getMoveName(g.board) == 'Kh5'
g.doMove(m)

m = Move.Move((3, 7), (6, 4), 3)
assert m.getMoveName(g.board) == 'Bxg5'
r, vec = g.getReward(m, p['mateReward'])

expected_r = p['gamma_exec']**2 * r + \
             p['gamma_exec']**3 * session.net.certainty * float(logit(session.net(vec, training=False)))

#   Confirm the traversal yields the same baseR
trav = Traversal.Traversal(game, session.net, p)
trav.traverse()
session.net.certainty = temp

print("Testing 'baseR' attribute of a traversal object (4)...")
misc.expect_equal(expected_r, trav.baseR, is_float=True)
