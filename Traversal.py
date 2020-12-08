import Game
import board_helper
import input_handling
import network_helper
import misc
import policy

import numpy as np
from scipy.special import expit, logit

class Traversal:
    def __init__(self, game, net, p):
        #   Parameters for traversal
        self.game = game
        self.net = net         
        self.nodeHops = 0
        self.pruneCuts = 0
        self.baseR = 0
        
        if p['policy'] == 'sampleMovesEG':
            self.policy = policy.sampleMovesEG
        else:
            self.policy = policy.sampleMovesSoft

        self.p = p
        #   A value not too far above the maximal number of node hops that can occur
        #   given the user's parameter specifications
        self.limit = 4 * p['breadth']**p['depth']
        

    #   From a base game, performs a depth-first tree traversal with the goal of
    #   approximating the reward along realistic move sequences. Breadth
    #   determines how many move options the engine explores from any node;
    #   reward is determined using a minimax search.
    def traverse(self):
        #   Handle the case where the game is already finished
        if self.game.gameResult != 17:
            #   Here it is assumed reward for the move resulting in the position given by
            #   self.game is already accounted for
            self.baseR = 0
            return

        p = self.p
        if p['mode'] >= 3:
            np.random.seed(0)
        else:
            np.random.seed()

        #   The maximum reward that could possibly be received from any position
        MAX_R = p['gamma_exec'] * p['mateReward']
        
        moves, fullMovesLen = self.policy(self.net, self.game, p)

        #   One element (node) on the stack is composed of the following:
        #   [moves, rewards, Game, reward up to this, alpha, beta]
        stack = [[moves, [], self.game, 0, -1 * MAX_R, MAX_R]]
        while len(stack) > 0:
            assert len(stack) <= p['depth'] + 1, "Tried to explore too far"
            assert self.nodeHops < self.limit, \
                   "Exceeded the number of node hops required to perform the entire traversal"
            if len(stack[-1][0]) > 0:   # if there are moves left to explore
                #   alpha-beta pruning
                if stack[-1][5] <= stack[-1][4]:
                    #   None of the remaining lines are relevant in this case
                    self.pruneCuts += len(stack[-1][0])
                    stack[-1][0] = []
                else:
                    #   Pop the first move and do it
                    g = stack[-1][2].copy()
                    r = g.getReward(stack[-1][0].pop(0),
                                    p['mateReward'],
                                    simple=True,
                                    copy=False)[0]
                    stack[-1][1].append(r)

                    #   If we aren't at the leaves, compute the next set of
                    #   moves/probs and add to the stack
                    if len(stack) < p['depth']:
                        if g.gameResult == 17:  
                            moves, fullMovesLen = self.policy(self.net, g, p)
                            stack.append([moves, [], g, stack[-1][3] + p['gamma_exec'] * r, stack[-1][4], stack[-1][5]])
                            self.nodeHops += 1
                        elif g.gameResult == 0:
                            self.nodeHops += 2
                        else:
                            #   Signal to stop branching here since we found a
                            #   mate (the strongest possible move)
                            stack[-1][0] = [] 
                            self.nodeHops += 2
                    #   At a leaf, we want to add the NN evaluation of the
                    #   position, scaled by our confidence in the NN, to make
                    #   sure rewards are not simply undone later in the game
                    elif self.net.certainty > p['minCertainty'] and g.gameResult == 17:
                        in_vec = g.toNN_vecs(every=False)[0]
                        stack[-1][1][-1] += self.net.certainty * p['gamma_exec'] * float(logit(self.net(in_vec, training=False)))

            else:   # otherwise hop down one node
                processNode(stack, self, p)


#   Descend one step of depth in the search after all moves for a node have been
#   explored. Pass reward down (minimax), along with alpha or beta as
#   appropriate.
def processNode(stack, trav, p):
    node = stack.pop()
    if node[2].whiteToMove:
        r = p['gamma_exec'] * max(node[1])
        
        #   Update beta if necessary
        if len(stack) > 0:
            stack[-1][5] = min(stack[-1][5], stack[-1][3] + r)

        
    else:
        r = p['gamma_exec'] * min(node[1])

        #   Update alpha if necessary
        if len(stack) > 0:
            stack[-1][4] = max(stack[-1][4], stack[-1][3] + r)

    #   Pass reward down if applicable
    if len(stack) > 0:
        stack[-1][1][-1] += r
    else:
        trav.baseR = r / p['gamma_exec']

    trav.nodeHops += 1
