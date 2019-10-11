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
    def __init__(self, game, net, p, isBase=True, reqMoves=[]):
        #   Parameters for traversal
        self.game = game
        self.net = net.copy()
        self.isBase = isBase         
        self.nodeHops = 0
        self.baseR = 0
        self.tData = []
        
        if p['policy'] == 'sampleMovesEG':
            self.policy = policy.sampleMovesEG
            self.policyVar = p['epsGreedy']
        else:
            self.policy = policy.sampleMovesSoft
            self.policyVar = p['curiosity']

        self.p = p.copy()
        if not isBase:
            self.p['tDepth'] -= 1
        #   A value not too far above the maximal number of node hops that can occur
        #   given the user's parameter specifications
        self.limit = 4 * p['breadth']**(p['tDepth'] + p['rDepth'])
        
        self.reqMoves = reqMoves
        self.numReqMoves = len(reqMoves) # since length of reqMoves will change 

    #   From a base game, performs a depth-first tree traversal with the goal of approximating
    #   the reward along realistic move sequences. Breadth determines how many move options the
    #   engine explores from any node; reward is a weighted sum along those paths, where weights
    #   are approximate probabilities that the engine would choose the respective paths. Reward
    #   is only considered if it occurs within *rDepth* moves from the given node. Generates a
    #   list of training examples (NN input, actual reward) of size
    #   (breadth ^ tDepth) / (breadth - 1)
    def traverse(self):
        p = self.p
        
        if self.numReqMoves > 0:
            reqMove = self.reqMoves.pop(0)
        else:
            reqMove = None
        moves, fullMovesLen = self.policy(self.net, self.game, p['breadth'], self.policyVar, p['mateReward'], reqMove)
        stack = [[moves, [], self.game, fullMovesLen]]
        while len(stack) > 0:
            assert len(stack) <= p['tDepth'] + p['rDepth'] + 1, "Tried to explore too far"
            assert self.nodeHops < self.limit, \
                   "Exceeded the number of node hops required to perform the entire traversal"
            if len(stack[-1][0]) > 0:   # if there are moves left to explore
                g = stack[-1][2].copy()
                g.doMove(stack[-1][0].pop(0))
                    
                #   Compute reward for the top move on the top slice of the stack, and add
                #   it to the top slice's cumulative total
                g0 = stack[-1][2]
                r = float(np.log(g.wValue * g0.bValue / (g.bValue * g0.wValue)))
                assert abs(r) < 1.3 * p['mateReward'], r
                stack[-1][1].append(r)

                #   If we aren't at the leaves, compute the next set of moves/probs and add
                #   to the stack
                if len(stack) <= p['tDepth'] + p['rDepth']:
                    if g.gameResult == 17:
                        if len(self.reqMoves) > 0 and self.numReqMoves >= len(stack):
                            reqMove = self.reqMoves.pop(0)
                        else:
                            reqMove = None
                        moves, fullMovesLen = self.policy(self.net, g, p['breadth'], self.policyVar, p['mateReward'], reqMove)
                        stack.append([moves, [], g, fullMovesLen])
                        self.nodeHops += 1
                    elif g.gameResult == 0:
                        #   Recompute reward received for the move, since the game ended
                        stack[-1][1][-1] = float(np.log(g0.bValue / g0.wValue))
                        self.nodeHops += 2
                    else:
                        #   A winning move should always be chosen; the engine should consider
                        #   the expected reward from last pos to be equal to the game result this pos
                        realBreadth = min(p['breadth'], stack[-1][3])
                        uncertainty = 1 - (stack[-1][3] - realBreadth) * (1 - p['alpha'])**realBreadth / stack[-1][3] 
                        stack[-1][1][-1] = g.gameResult * p['mateReward'] / uncertainty
                        stack[-1][0] = [] # for speed, stop branching here since we found a mate
                        self.nodeHops += 2
                #   At a leaf, we want to add the NN evaluation of the position, scaled by our
                #   confidence in the NN, to make sure rewards are not simply undone later in the game
                else:
                    stack[-1][1][-1] += p['alpha'] * float(logit(self.net.feedForward(g.toNN_vecs()[0])))

            else:   # otherwise hop down one node
                self.nodeHops += 1
                node = stack.pop()
                r = processNode(node, p['breadth'], p['clarity'], p['alpha'])
                #   If the node is sufficiently shallow, include it as training data
                if len(stack) <= p['tDepth']:
                    in_vecs = node[2].toNN_vecs()
                    self.tData.append((in_vecs[0], expit(r)))
                    self.tData.append((in_vecs[1], expit(-1*r)))
                #   Pass reward down if applicable
                if len(stack) > 0:
                    stack[-1][1][-1] += r
                elif len(stack) == 0 and not self.isBase:
                    self.baseR = r
    
#   Given a list (element on the stack during traversal), return the expected value of
#   the reward from the position associated with that element.
def processNode(node, breadth, clarity, alpha):
    #   Take softmax of rewards (interpreted as probabilities of playing them)
    #r = np.array(node[1]) * (2 * node[2].whiteToMove - 1)
    #softR = np.exp(clarity * r)
    #softR = softR / np.sum(softR)
    
    #   Sum of "probabilities" * rewards (aka the expected reward from this node),
    #   scaled by uncertainty (see readme for explanation of math)
    realBreadth = min(breadth, node[3])
    uncertainty = 1 - (node[3] - realBreadth) * (1 - alpha)**realBreadth / node[3]
    if node[2].whiteToMove:
        r = max(node[1])
    else:
        r = min(node[1])
    #return float(softR @ np.array(node[1]).T) * uncertainty
    return float(r * uncertainty)

def full_high_R(net):
    p = input_handling.readConfig(1)
                    
    bestMoves = initializeGame(net, True)[0]
    
    g = Game.Game(quiet = True)
    movesIn = len(bestMoves) - p['rDepth'] - p['tDepth']
    for i in range(movesIn):
        g.doMove(bestMoves[i])

    tData = do_multicore(net, g, bestMoves[movesIn:])

    return tData

def full_broad(net):
    p = input_handling.readConfig(1)
    gBuffer = p['gBuffer']
    tDepth = p['tDepth']
    rDepth = p['rDepth']
    numStarts = p['baseBreadth']

    #   Initialize a base game, from which a start node will be chosen
    firstTry = True
    while firstTry or len(bestMoves) <= 2*gBuffer + numStarts * (rDepth + tDepth):
        gBuffer = p['gBuffer']
        numStarts = p['baseBreadth']
        if not firstTry:
            assert p['epsilon'] > 0, \
                "Fatal error: game was not long enough to fit requested traversal depth even with gBuffer = 0." +\
                "Cannot remake a longer game since epsilon = 0 (consider raising epsilon or lowering baseBreadth)."
            
            print("Remaking game, since the last one wasn't long enough even w/ gBuffer=0...")
            
        firstTry = False
        bestMoves, captures = initializeGame(net, False, True)

        #   Relax parameter choices if necessary to fit the traversals. First, decrease gBuffer to a minimum
        #   of 0. Then reduce the number of traversals ran to a minimum of 1.
        while(len(bestMoves) <= 2*gBuffer + numStarts*(rDepth + tDepth) and gBuffer > 0 and numStarts > 1):
            if gBuffer > 0:
                gBuffer = int(gBuffer / 2)
            else:
                numStarts -= 1

    #   Notify user of how parameters were relaxed   
    if gBuffer < p['gBuffer']:
        print("Warning: needed to trim gBuffer to", gBuffer, "to fit traversals at requested depth")
        if numStarts < p['baseBreadth']:
            print("Warning: also needed to reduce to", numStarts, "traversals to fit each without overlap")

    #   Split game into "blocks" of moves, and randomly choose a start index within each block. This allows potential
    #   for overlapping traversals, but this should be uncommon and still have little effect on training power.
    blockSize = int((len(bestMoves) - 2*gBuffer) / numStarts)
    assert blockSize > 0, str(len(bestMoves) - 2*gBuffer) + " " + str(numStarts)
    captLists = [np.cumsum(np.array(captures[i*blockSize+gBuffer: (i+1)*blockSize+gBuffer])) for i in range(numStarts)]
    gStarts = []
    for i, cList in enumerate(captLists):
        if cList[-1] == 0:
            print("Warning: no captures were possible during the entire game for block", i, ".")
            gStarts.append(np.random.randint(gBuffer+i*blockSize, gBuffer+(i+1)*blockSize))
        else:
            gStarts.append(misc.sampleCDF(cList/ cList[-1])[0] + gBuffer + i*blockSize)

    #   Prepare the Traversal objects
    trav_objs = []
    for i, s in enumerate(gStarts):
        print("Game ", i, " will start at move ", s, ", with ", captures[s], " captures possible.", sep="")
        
        #   Do a game up to the specified start
        g = Game.Game(quiet=True)
        for j in range(max(0, s-1)):
            g.doMove(bestMoves[j])
        
        trav_objs.append(Traversal(g, net, p))

    #   Do the traversals and return the training examples
    print("--------------------------------------")
    print("Beginning broad multicore traversal...")
    print("--------------------------------------")
    
    pool = Pool()
    res_objs = pool.map_async(per_thread_job, trav_objs).get()
    pool.close()
    tData = []
    nodeHops = 0
    for ob in res_objs:
        tData += ob.tData
        nodeHops += ob.nodeHops
    print("Done. Hopped between nodes ~" + str(nodeHops) + " times and " +
          "generated " + str(len(tData)) + " training examples.")

    return tData
        
def full_low_R(net):
    p = input_handling.readConfig(1)
    gBuffer = p['gBuffer']
    tDepth = p['tDepth']
    rDepth = p['rDepth']

    #   Initialize a base game, from which a start node will be chosen
    firstTry = True
    while firstTry or len(bestMoves) < 2*gBuffer + rDepth + tDepth:
        gBuffer = p['gBuffer']
        if not firstTry:
            assert p['epsilon'] > 0, \
                "Fatal error: game was not long enough to fit requested traversal depth even with gBuffer = 0." +\
                "Cannot remake a longer game since epsilon = 0 (consider raising epsilon)."
            
            print("Remaking game, since the last one wasn't long enough even w/ gBuffer=0...")
            
        firstTry = False
        bestMoves, captures = initializeGame(net, False, True)

        #   If needed, half gBuffer recursively to fit traversals. rDepth, tDepth, and epsilon
        #   are hard requirements, and currently the program will exit if a sufficiently long game cannot be
        #   made even with gBuffer = 0.
        while(len(bestMoves) < 2*gBuffer + rDepth + tDepth and gBuffer > 0):
            gBuffer = int(gBuffer / 2)          
        
    if gBuffer < p['gBuffer']:
        print("Warning: needed to trim gBuffer to", gBuffer, "to fit traversals at requested depth")

    #   Now select the game start randomly with chance proportional to the number of captures available
    #   from that start position
    upperBound = len(bestMoves) - (gBuffer + rDepth + tDepth)
    assert upperBound > gBuffer, "Didn't trim game or warn user despite inadequate number of moves"
    cumDistr = np.cumsum(np.array(captures[gBuffer: upperBound+1]))
    if cumDistr[-1] == 0:
        print("Warning: no captures were possible during the entire game.")
        gStart = np.random.randint(gBuffer, upperBound+1)
    else:
        gStart = max(0, misc.sampleCDF(cumDistr / cumDistr[-1])[0] + gBuffer - 1)

    #   The [gStart+1]th move should tend to be a choice among many potential captures
    print("Game randomly chosen to start at move", gStart+1, "where there are", captures[gStart+1], "captures.")

    #   Initialize game, gStart moves into the base game
    g = Game.Game()
    g.quiet = True
    for i in range(gStart):
        g.doMove(bestMoves[i])

    tData = do_multicore(net, g)

    return tData

def per_thread_job(trav_obj):
    trav_obj.traverse()
    return trav_obj

def initializeGame(net, strict, trackCaptures=False):
    print("-------------------------")
    print("Initializing base game...")
    print("-------------------------")

    p = input_handling.readConfig(1)

    firstTry = True
    while(firstTry or (strict and game.gameResult == 0)):
        firstTry = False
        game = Game.Game(quiet=False)
        bestMoves = list()
        captures = list()
        while (game.gameResult == 17):
            #   Find the best legal move
            legalMoves = board_helper.getLegalMoves(game)
            bestMove = policy.getBestMoveEG(game, legalMoves, net, p['epsilon'], p['mateReward'])
            bestMoves.append(bestMove)

            #   Track number of possible captures
            if trackCaptures:
                captCount = sum([m.isCapture(game.board) for m in legalMoves])
            else:
                captCount = 0       
            captures.append(captCount)

            game.doMove(bestMove)

    return (bestMoves, captures)

def do_multicore(net, g, reqMoves=[]):
    p = input_handling.readConfig(1)
    
    #   Build the list of Traversal objects which will be operated on using multiple cores
    if len(reqMoves) == 0:
        reqMove = None
    else:
        reqMove = reqMoves[0]
    m, fullMovesLen = sampleMoves(net, g, p['baseBreadth'], p['curiosity'], p['mateReward'], reqMove)

    r = 0
    trav_objs = []
    for i, mov in enumerate(m):
        g2 = g.copy()
        g2.doMove(mov)
        if g2.gameResult == 17:
            #   The "required moves" to pass to 'traverse' follow from only the first move from depth 0
            if i == 0 and len(reqMoves) > 0:
                reqM = reqMoves[1:]
            else:
                reqM = []
            trav_objs.append(Traversal(g2, net, p, False, reqM))
        elif not g2.gameResult == 0:
            #   Note that we continue exploring the tree (unlike the traverse() method) from other nodes
            r = g2.gameResult * p['mateReward']

    print("--------------------------------")
    print("Beginning multicore traversal...")
    print("--------------------------------")
    
    pool = Pool()
    res_objs = pool.map_async(per_thread_job, trav_objs).get()
    pool.close()
    tData = []
    nodeHops = 0
    for ob in res_objs:
        tData += ob.tData
        nodeHops += ob.nodeHops
    print("Done. Hopped between nodes ~" + str(nodeHops) + " times and " +
          "generated " + str(len(tData)) + " training examples.")    

    #   Add the base node as a training example ------
    #   Again, a linear combination of received reward * p(move) and reward / num of possible moves,
    #   weighted by alpha, for each move. Each rew is summed up and scaled by the fraction of moves
    #   explored (controlling for uncertainty)
    if r == 0:  # in other words, if the base node was not connected in one move to a win/ loss
        rVec = np.array([ob.baseR for i in res_objs])
        softR = np.exp(rVec * (2 * g.whiteToMove - 1) * p['clarity'])
        softR = softR / np.sum(softR)

        realBreadth = min(p['baseBreadth'], fullMovesLen)
        uncertainty = 1 - (fullMovesLen - realBreadth) * (1 - p['alpha'])**realBreadth / fullMovesLen
        r = float(softR @ rVec.T) * uncertainty
    
    NN_inputs = g.toNN_vecs()
    #assert abs(expit(r) - 0.5) < 0.5, r
    tData.append((NN_inputs[0], expit(r)))
    tData.append((NN_inputs[1], expit(-1*r)))

    return tData

