import numpy as np

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
        'policyFun': "sampleMovesEG",
        'evalFun': policy.getEvalsDebug,
        'mode': 3,
        'epsSearch': 0
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
        'policyFun': "sampleMovesEG",
        'evalFun': policy.getEvalsDebug,
        'mode': 3,
        'epsSearch': 0
    }

    trav = Traversal.Traversal(game, None, p, fake_evals = True)
    trav.traverse(num_steps = 4)
    assert len(trav.stack) == 0, len(trav.stack)

    #   The best move is 'Ra7', where the only reward is the fake eval of 0.1
    assert abs(trav.baseR - 0.1) < tol
    assert trav.bestLine == ['Ra7'], trav.bestLine
