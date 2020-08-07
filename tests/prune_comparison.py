import sys
sys.path.append('../')

import input_handling
import Game
import policy
import board_helper
import old_traversal
import Traversal
import Network

from multiprocessing import Pool
import numpy as np
import time

def getBestMoveTreeEG(net, game, p, pool=None):
    if np.random.uniform() < p['epsGreedy']:
        legalMoves = board_helper.getLegalMoves(game)
        return legalMoves[np.random.randint(len(legalMoves))]
    else:
        #   Traversals are started at positions resulting from testing moves from
        #   the current position; this test constitutes a step of depth
        p_copy = p.copy()
        p_copy['depth'] -= 1
        p = p_copy
        
        moves, fullMovesLen = policy.sampleMovesEG(net, game, p)
        
        rTemp = np.zeros(len(moves))
        if pool != None:
            trav_objs = []
            for i, m in enumerate(moves):
                rTemp[i] = game.getReward(m, p['mateReward'], True)[0]

                g = game.copy()
                g.quiet = True
                g.doMove(m)
                trav_objs.append(old_traversal.Traversal(g, net, p))

            #   Perform the traversals in parallel to return the index of the first
            #   move from this position of the most rewarding move sequence explored
            res_objs = pool.map(old_traversal.per_thread_job, trav_objs)

            baseRs = np.array([ob.baseR for ob in res_objs])
            rTemp += p['gamma_exec'] * baseRs
        else:  
            for i, m in enumerate(moves):
                g = game.copy()
                g.quiet = True
                g.doMove(m)
                        
                trav = old_traversal.Traversal(g, net, p)
                trav.traverse()
                rTemp[i] = game.getReward(m, p['mateReward'], True)[0] + p['gamma_exec'] * trav.baseR
                #rTemp[i] = game.getReward(m, p['mateReward'], True)[0] + certainty * trav.baseR

        if game.whiteToMove:
            bestMove = moves[np.argmax(rTemp)]
        else:
            bestMove = moves[np.argmin(rTemp)]

        return bestMove

#######################################
#   Main
#######################################

p = input_handling.readConfig(3, '../config.txt')
p.update(input_handling.readConfig(1, '../config.txt'))
p['epsGreedy'] = 0
p['epsSearch'] = 0
#net, dummy, dummy2 = Network.load('../nets/res3', lazy=True)
net = Network.Network([20, 20, 20, 1], 2, 1)
#pool = Pool()

#--------------------------------------
#   Old method
#--------------------------------------

#old_time = 390.86
print("Starting old traversal...")
old_time = time.time()
game = Game.Game(quiet=False)

while (game.gameResult == 17):
    bestMove = getBestMoveTreeEG(net, game, p)#, pool)
    game.doMove(bestMove)

old_time = time.time() - old_time
game.toPGN(filename='../visualization/prune_comparison/old_method.pgn')

#--------------------------------------
#   New method
#--------------------------------------

print("Starting new traversal...")
new_time = time.time()
game = Game.Game(quiet=False)

while (game.gameResult == 17):
    bestMove = policy.getBestMoveTreeEG(net, game, p)#, pool=pool)
    game.doMove(bestMove)

new_time = time.time() - new_time
game.toPGN(filename='../visualization/prune_comparison/new_method.pgn')

#pool.close()

print("Done.")
print("Old time:", old_time)
print("New time: ", new_time, '(', round(100 * new_time / old_time, 2), "%)")
