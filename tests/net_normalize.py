import sys
sys.path.append('../')
from scipy.special import expit, logit
from multiprocessing import Pool
import numpy as np

import q_learn
import input_handling
import Network
import Game
import policy

repeats = 10
net = Network.load('../nets/8deep7', True)[0]

##########################################################
#   Estimate variance of observed vs. predicted reward
##########################################################

p = input_handling.readConfig(1, '../config.txt')
p.update(input_handling.readConfig(3, '../config.txt'))

#   Run asynchronous data generation
print("Producing training data (" + str(p['baseBreadth']) + " tasks)...")
pool = Pool()
inList = [(net, p) for i in range(p['baseBreadth'])]
thread_data = pool.starmap_async(q_learn.generateExamples, inList).get()
#pool.close()

#   Collect each process's results (data) into a single list
data = [[],[],[]]
for x in thread_data:
    for i in range(3):
        data[i] += x[i]
print("Done. Generated " + str(sum([len(x) for x in data])) + " training examples.\n")
        
#   Get only the originally generated examples (do not include augmented data)
print("Filtering to only non-augmented data and determining ratio for scaling certainty...")
origData = [data[0][i] for i in range(len(data[0])) if i % 2 == 0] + \
           [data[1][i] for i in range(len(data[1])) if i % 4 == 0] + \
           [data[2][i] for i in range(len(data[2])) if i % 16 == 0]

#   Form vectors of expected and actual rewards received
expRew = logit(np.array([net.feedForward(x[0]) for x in origData]).flatten())
actRew = logit(np.array([x[1] for x in origData]).flatten())

net_norm = net.copy()
net_norm.certainty *= float(np.std(actRew) / np.std(expRew))
print("Done. Running games using normalized vs original network confidences...")

###############################################################
#   getBestMoveEG w/ normalized vs. original net confidences
###############################################################

results = 0     # how many times "better" policy wins
for i in range(repeats):
    game = Game.Game()
    while (game.gameResult == 17):
        #   This ensures there isn't a bias for white winning, for example
        if (game.whiteToMove + i) % 2 == 1:
            bestMove = policy.getBestMoveTreeEG(net_norm, game, p, pool)
        else:
            bestMove = policy.getBestMoveTreeEG(net, game, p, pool)
        game.doMove(bestMove)
        
    #   Flip the game result if the supposedly better policy is being played
    #   by black each turn instead of white
    results += game.gameResult * (1 - 2 * (i % 2))

pool.close()
print("Average game result for normalized EG vs. original EG policy:", round(results/ repeats, 4))
