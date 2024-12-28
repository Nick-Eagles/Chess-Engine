import numpy as np
from pyhere import here
import sys

sys.path.append(str(here()))
import policy
import Traversal
import Game
import board_helper

#   Tolerance for floating-point differences when checking equality
tol = 0.00001

def test_traverse():
    ############################################################################
    #   First position
    ############################################################################

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
    game.currentlyCheck = True
    game.updateValues()

    p = {
        'depth': 3,
        'breadth': 4,
        'gamma_exec': 0.9,
        'mateReward': 3,
        'policyFun': "sampleMovesStatic",
        'evalFun': policy.getEvalsDebug,
        'mode': 3
    }

    trav = Traversal.Traversal(game, None, p, fake_evals = True)
    trav.traverse(num_steps = 6)

    assert len(trav.stack) == 3, len(trav.stack)

    #   Kxh3 is the only legal move for black
    assert trav.stack[0]['move_names'] == [['Kxh3']], trav.stack[0]['move_names']
    assert len(trav.stack[0]['moves']) == 0, len(trav.stack[0]['moves'])

    #   Then Kh1 is the only legal move for white
    assert trav.stack[1]['move_names'] == [['Kh1']], trav.stack[1]['move_names']
    assert len(trav.stack[1]['moves']) == 0, len(trav.stack[1]['moves'])

    #   These are the first 4 moves (by eval, which is alphabetical by move
    #   name) that black can play
    assert trav.stack[2]['move_names'] == [['Kg3'], ['Kg4'], ['Rf1'], ['Rf2']], trav.stack[2]['move_names']
    assert len(trav.stack[2]['moves']) == 0, len(trav.stack[2]['moves'])

    #   At the end of the traversal, the best line involves a queen capture and
    #   mate by black, which should have this reward
    expected_reward = np.log(4 * 12 / (13 * 12)) - \
        p['gamma_exec']**2 * p['mateReward']

    #   Completing the tree search
    trav.traverse(num_steps = 3)
    assert len(trav.stack) == 0, len(trav.stack)
    assert abs(trav.baseR - expected_reward) < tol
    assert trav.bestLine == ['Kxh3', 'Kh1', 'Rf1'], trav.bestLine
    assert trav.bestMove.getMoveName(game) == 'Kxh3', trav.bestMove.getMoveName(game)

    ############################################################################
    #   Second position
    ############################################################################

    game = Game.Game()
    game.board = [
        [0, 0, 0, 0, 0, 0, 0, -6],
        [0, 0, 0, 0, 1, 1, 4, 0],
        [0, 0, 0, 0, 1, 6, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    game.invBoard = board_helper.invert(game.board)
    game.wPieces = [6, 0, 0, 0, 1, 0]
    game.bPieces = [0, 0, 0, 0, 0, 0]
    game.canW_K_Castle = False
    game.canW_Q_Castle = False
    game.canB_K_Castle = False
    game.canB_Q_Castle = False
    game.moveNum = 20
    game.updateValues()

    p = {
        'depth': 1,
        'breadth': 2,
        'gamma_exec': 0.8,
        'mateReward': 3,
        'policyFun': "sampleMovesStatic",
        'evalFun': policy.getEvalsDebug,
        'mode': 3
    }

    trav = Traversal.Traversal(game, None, p, fake_evals = True)
    trav.traverse(num_steps = 4)
    assert len(trav.stack) == 0, len(trav.stack)

    #   The best move is 'Ra7', where the only reward is the fake eval of 0.1
    assert abs(trav.baseR - 0.1) < tol
    assert trav.bestLine == ['Ra7'], trav.bestLine
    assert trav.bestMove.getMoveName(game) == 'Ra7', trav.bestMove.getMoveName(game)

    ############################################################################
    #   Empirically test alpha-beta pruning
    ############################################################################

    #---------------------------------------------------------------------------
    #   Position 1
    #---------------------------------------------------------------------------

    game = Game.Game()
    game.board = [
        [4, 0, 1, 0, 0, -1, 0, -4],
        [0, 0, 0, 1, 0, 0, -1, 0],
        [0, 1, -2, 0, 0, 0, -5, 0],
        [0, 2, 3, 2, -1, 0, -3, 0],
        [4, 0, 0, 0, 1, -1, -3, 0],
        [0, 1, 0, 0, 0, 0, -1, -4],
        [6, 1, 0, 5, 0, 0, -1, -6],
        [0, 1, 0, 0, 0, -1, 0, 0]
    ]
    game.invBoard = board_helper.invert(game.board)
    game.wPieces = [7, 2, 1, 0, 2, 1]
    game.bPieces = [7, 1, 1, 1, 2, 1]
    game.canW_K_Castle = False
    game.canW_Q_Castle = False
    game.canB_K_Castle = False
    game.canB_Q_Castle = False
    game.moveNum = 18
    game.updateValues()

    p = {
        'depth': 4,
        'breadth': 4,
        'gamma_exec': 0.7,
        'mateReward': 4,
        'policyFun': "sampleMovesStatic",
        'evalFun': policy.getEvalsDebug,
        'mode': 3
    }

    trav1 = Traversal.Traversal(game, None, p, fake_evals = True, prune = False)
    trav1.traverse()
    trav2 = Traversal.Traversal(game, None, p, fake_evals = True, prune = True)
    trav2.traverse()
    assert trav1.bestLine == trav2.bestLine
    assert trav1.bestMove.equals(trav2.bestMove)
    assert trav1.baseR == trav2.baseR
    assert trav1.numLeaves > trav2.numLeaves

    #---------------------------------------------------------------------------
    #   Position 2
    #---------------------------------------------------------------------------

    game = Game.Game()
    game.board = [
        [0, 0, 0, 3, 0, -6, 0, 0],
        [0, 0, 1, 0, -3, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 1, 2, 0, 0, 0, 0],
        [0, 1, 1, 6, 1, 1, 0, 0],
        [0, 0, 4, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    game.invBoard = board_helper.invert(game.board)
    game.wPieces = [8, 1, 1, 0, 1, 0]
    game.bPieces = [1, 0, 1, 0, 0, 0]
    game.canW_K_Castle = False
    game.canW_Q_Castle = False
    game.canB_K_Castle = False
    game.canB_Q_Castle = False
    game.moveNum = 35
    game.updateValues()

    p = {
        'depth': 4,
        'breadth': 6,
        'gamma_exec': 0.95,
        'mateReward': 2.6,
        'policyFun': "sampleMovesStatic",
        'evalFun': policy.getEvalsDebug,
        'mode': 3
    }

    trav1 = Traversal.Traversal(game, None, p, fake_evals = True, prune = False)
    trav1.traverse()
    trav2 = Traversal.Traversal(game, None, p, fake_evals = True, prune = True)
    trav2.traverse()
    assert trav1.bestLine == trav2.bestLine
    assert trav1.bestMove.equals(trav2.bestMove)
    assert trav1.baseR == trav2.baseR
    assert trav1.numLeaves > trav2.numLeaves

    #---------------------------------------------------------------------------
    #   Position 3
    #---------------------------------------------------------------------------

    game = Game.Game()
    game.board = [
        [0, 6, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 2, -6]
    ]
    game.invBoard = board_helper.invert(game.board)
    game.wPieces = [1, 1, 1, 0, 0, 0]
    game.bPieces = [2, 0, 0, 0, 1, 0]
    game.canW_K_Castle = False
    game.canW_Q_Castle = False
    game.canB_K_Castle = False
    game.canB_Q_Castle = False
    game.moveNum = 41
    game.updateValues()

    p = {
        'depth': 5,
        'breadth': 4,
        'gamma_exec': 0.75,
        'mateReward': 2.8,
        'policyFun': "sampleMovesStatic",
        'evalFun': policy.getEvalsDebug,
        'mode': 3
    }

    #   The optimal move sequence here is intentionally put as the path to the last
    #   leaf, so pruning should not occur (and would interfere with calculation of
    #   optimal sequence otherwise)
    trav1 = Traversal.Traversal(game, None, p, fake_evals = True, prune = False)
    trav1.traverse()
    trav2 = Traversal.Traversal(game, None, p, fake_evals = True, prune = True)
    trav2.traverse()
    assert trav1.bestLine == trav2.bestLine
    assert trav1.bestMove.equals(trav2.bestMove)
    assert trav1.baseR == trav2.baseR
    assert trav1.numLeaves == trav2.numLeaves
