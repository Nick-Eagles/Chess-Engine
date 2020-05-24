import Game
import board_helper
import input_handling
import network_helper
import misc
import policy

import numpy as np
import random
from scipy.special import expit, logit
from multiprocessing import Pool

class Traversal:
    def __init__(self, game, net, p):
        #   Parameters for traversal
        self.game = game
        self.net = net         
        self.nodeHops = 0
        self.baseR = 0
        
        if p['policy'] == 'sampleMovesEG':
            self.policy = policy.sampleMovesEG
        else:
            self.policy = policy.sampleMovesSoft

        self.p = p
        #   A value not too far above the maximal number of node hops that can occur
        #   given the user's parameter specifications
        self.limit = 4 * p['breadth']**p['depth']
        

    #   From a base game, performs a depth-first tree traversal with the goal of approximating
    #   the reward along realistic move sequences. Breadth determines how many move options the
    #   engine explores from any node; reward is determined using a minimax search. 
    def traverse(self):
        #   Handle the case where the game is already finished
        if self.game.gameResult != 17:
            #   Here it is assumed reward for the move resulting in the position given by
            #   self.game is already accounted for
            self.baseR = 0
            return
        
        np.random.seed()
        p = self.p
        
        moves, fullMovesLen = self.policy(self.net, self.game, p)
        stack = [[moves, [], self.game, fullMovesLen]]
        while len(stack) > 0:
            assert len(stack) <= p['depth'] + 1, "Tried to explore too far"
            assert self.nodeHops < self.limit, \
                   "Exceeded the number of node hops required to perform the entire traversal"
            if len(stack[-1][0]) > 0:   # if there are moves left to explore
                g = stack[-1][2].copy()
                g.doMove(stack[-1][0].pop(0))
                    
                #   Compute reward for the top move on the top slice of the stack, and add
                #   it to the top slice's cumulative total
                g0 = stack[-1][2]
                r = GetReward(g0, g, p)
                stack[-1][1].append(r)

                #   If we aren't at the leaves, compute the next set of moves/probs and add
                #   to the stack
                if len(stack) < p['depth']:
                    if g.gameResult == 17:
                        moves, fullMovesLen = self.policy(self.net, g, p)
                        stack.append([moves, [], g, fullMovesLen])
                        self.nodeHops += 1
                    elif g.gameResult == 0:
                        self.nodeHops += 2
                    else:
                        #   Signal to stop branching here since we found a mate
                        #   (the strongest possible move)
                        stack[-1][0] = [] 
                        self.nodeHops += 2
                #   At a leaf, we want to add the NN evaluation of the position, scaled by our
                #   confidence in the NN, to make sure rewards are not simply undone later in the game
                elif self.net.certainty > p['minCertainty']:
                    stack[-1][1][-1] += self.net.certainty * p['gamma_exec'] * float(logit(self.net.feedForward(g.toNN_vecs(every=False)[0])))

            else:   # otherwise hop down one node
                self.nodeHops += 1
                r = processNode(stack.pop(), p)

                #   Pass reward down if applicable
                if len(stack) > 0:
                    stack[-1][1][-1] += r
                else:
                    self.baseR = r


def GetReward(g0, g, p):
    if abs(g.gameResult) == 1:
        return g.gameResult * p['mateReward']
    elif g.gameResult == 0:
        return float(np.log(g0.bValue / g0.wValue))
    else:
        return float(np.log(g.wValue * g0.bValue / (g.bValue * g0.wValue)))


#   Given a list (element on the stack during traversal), return the expected value of
#   the reward from the position associated with that element.
def processNode(node, p):
    if node[2].whiteToMove:
        r = max(node[1])
    else:
        r = min(node[1])

    return r * p['gamma_exec']


def per_thread_job(trav_obj):
    trav_obj.traverse()
    return trav_obj
