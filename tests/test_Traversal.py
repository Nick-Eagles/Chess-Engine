import numpy as np

import policy
import Traversal
import Game
import board_helper

#   Tolerance for floating-point differences when checking equality
tol = 0.00001

#   Manually produce a particular position
game = Game.Game()
game.board = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, -4, -1, 0, 0, 0, 0],
    [6, 0, -6, 0, 0, 0, 0, 0],
    [0, 0, 5, -1, 0, 0, 0, 0]
]
game.invBoard = board_helper.invert(game.board)
game.wPieces = [0, 0, 0, 0, 0, 1]
game.bPieces = [3, 0, 0, 0, 1, 0]
game.canW_K_Castle = False
game.canW_Q_Castle = False
game.canB_K_Castle = False
game.canB_Q_Castle = False
game.whiteToMove = False
game.updateValues()

p = {}
p['depth'] = 3
p['breadth'] = 4
p['gamma_exec'] = 0.9
p['mateReward'] = 3
p['policyFun'] = "sampleMovesEG"
p['evalFun'] = policy.getEvalsDebug

trav = Traversal.Traversal(game, None, p)
trav.traverse(num_steps = 7)

assert len(trav.stack) == 3, len(trav.stack)

#   Kxh3 is the only legal move for black
assert trav.stack[0]['move_names'] == ['Kxh3']
assert len(trav.stack[0]['moves']) == 0, len(trav.stack[0]['moves'])

#   Then Kh1 is the only legal move for white
assert trav.stack[1]['move_names'] == ['Kh1']
assert len(trav.stack[1]['moves']) == 0, len(trav.stack[1]['moves'])

#   These are the first 5 moves (by eval, which is alphabetical by move name)
#   that black can play. Only 6 moves are possible
assert trav.stack[2]['move_names'] == ['e7', 'Kg3', 'Kg4', 'Rf7', 'Rf8'], trav.stack[2]['move_names']
assert len(trav.stack[2]['moves']) == 1, len(trav.stack[2]['moves'])

#   At the end of the traversal, the best line involves a queen capture and mate
#   by black, which should have this reward
expected_reward = p['gamma_exec'] * np.log(4 * 12 / (13 * 12)) - \
    p['gamma_exec']**3 * p['mateReward']

#   Completing the tree search
trav.traverse(num_steps = 4)
assert len(trav.stack) == 0, len(trav.stack)
assert abs(trav.baseR - expected_reward) < tol 
