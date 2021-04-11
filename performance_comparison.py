#   Utilities for comparing the relative performance of models in terms of
#   "chess skill"

import Session
import input_handling
import Game
import policy

import time

#   Return a 2-tuple of integers: first, the relative score from
#   playing n games with a particular model, using two different parameter sets
#   p1 and p2; second, the ratio of total elapsed times for each parameter set
#
#   A win for the first set yields a value of 1; a loss yields -1, and a draw
#   yields 0
def compare_p(model, p1, p2, n, move_limit=100, verbose=False):
    score = 0
    elapsed1 = 0
    elapsed2 = 0
    
    for i in range(n):
        game = Game.Game(quiet= not verbose)

        if verbose:
            print('Starting game ', i+1, '!', sep='')

        #   Perform a single game
        move_num = 0
        while (game.gameResult == 17 and move_num < move_limit):
            #   Colors switch every move and every game
            start = time.time()
            if (move_num + i) % 2 == 0:
                bestMove = policy.getBestMoveTreeEG(model, game, p1)
                elapsed1 += time.time() - start
            else:
                bestMove = policy.getBestMoveTreeEG(model, game, p2)
                elapsed2 += time.time() - start

            #print(bestMove.toString())
            game.doMove(bestMove)
            move_num += 1

        if move_num < move_limit:
            #   Accounts for each "player" switching colors every game
            scalar = 1 - 2 * (i % 2)
            
            score += game.gameResult * scalar
            
            if verbose:
                print('Game completed. Score:', game.gameResult * scalar)
        elif verbose:
            print('Game considered drawn (too long).')

    return (score, elapsed1 / elapsed2)


print('Loading model...')
session = Session.Session([], [])
session.Load('nets/tf_ex_compare')

print('Loading configuration...')      
p1 = input_handling.readConfig()
p2 = p1.copy()

p1['depth'] = 3
p1['breadth'] = 8

p2['depth'] = 2
p2['breadth'] = 12

print('Playing games...')
results = compare_p(session.net, p1, p2, 10, verbose=True)
print(results)
