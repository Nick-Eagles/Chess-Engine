import policy

import numpy as np
import tensorflow as tf

class Traversal:
    def __init__(self, game, net, p, fake_evals = False, prune = True):
        #   Parameters for monitoring extent and results of search
        self.numLeaves = 0
        self.prune = prune
        self.bestLine = []

        #   The neural network to use. If fake_evals, simulate small evaluations
        #   at leaves in the tree search (for testing). Otherwise use the neural
        #   network
        self.net = net
        self.fakeEvals = fake_evals

        self.baseR = 0
        self.bestMove = None
        self.policy = getattr(policy, p['policyFun'])
        self.p = p

        #   The maximum reward that could possibly be received from any position
        MAX_R = p['gamma_exec'] * p['mateReward']

        #   Initialize the stack, each element of which represents a game
        #   position. Adding an element to the stack will represent exploring
        #   one position deeper at the current position in the tree.
        #   One element (node) on the stack is composed of the following:
        #   [moves, rewards, Game, reward up to this, alpha, beta, list of
        #    NN inputs, move names]
        moves = self.policy(self.net, game, p)
        self.stack = [
            {
                'moves': moves, 'rewards': [], 'game': game,
                'prev_reward': 0, 'alpha': -1 * MAX_R, 'beta': MAX_R,
                'nn_inputs': [], 'move_names': [], 'moves_done': []
            }
        ]
        
    #   From a base game, performs a depth-first tree traversal with the goal of
    #   approximating the reward along realistic move sequences. Reward is
    #   determined using a minimax search.
    #
    #   num_steps: int or None. If None, perform the entire traversal. If an
    #       int, perform only that number of steps (useful for unit testing)
    def traverse(self, num_steps = None):
        p = self.p
        stack = self.stack

        #   Handle the case where the game is already finished
        if stack[0]['game'].gameResult != 17:
            return

        if p['mode'] >= 3:
            np.random.seed(0)
        else:
            np.random.seed()
        
        step = 0
        while (len(stack) > 0) and ((num_steps is None) or (step < num_steps)):
            step += 1
            if len(stack[-1]['moves']) > 0:   # if there are moves left to explore
                #   alpha-beta pruning
                if self.prune and (stack[-1]['beta'] <= stack[-1]['alpha']):
                    #   None of the remaining lines are relevant in this case
                    stack[-1]['moves'] = []
                else:
                    #   Pop the first move and do it
                    g = stack[-1]['game'].copy()
                    m = stack[-1]['moves'].pop(0)
                    stack[-1]['move_names'].append([m.getMoveName(g)])
                    stack[-1]['moves_done'].append(m)
                    r = g.getReward(
                        m, p['mateReward'], simple=True, copy=False
                    )[0]
                    stack[-1]['rewards'].append(r)
                    

                    #   If we aren't at the leaves, compute the next set of
                    #   moves/probs and add to the stack
                    if len(stack) < p['depth']:
                        if g.gameResult == 17:  
                            moves = self.policy(self.net, g, p)
                            stack.append(
                                {
                                    'moves': moves,
                                    'rewards': [],
                                    'game': g,
                                    'prev_reward': stack[-1]['prev_reward'] + \
                                        p['gamma_exec'] * r,
                                    'alpha': stack[-1]['alpha'],
                                    'beta': stack[-1]['beta'],
                                    'nn_inputs': [],
                                    'move_names': [],
                                    'moves_done': []
                                }
                            )
                        elif abs(g.gameResult) == 1:
                            #   Signal to stop branching here since we found a
                            #   mate (the strongest possible move)
                            stack[-1]['moves'] = []

                    #   At a leaf, we want to add the NN evaluation of the
                    #   position to make sure rewards are not simply undone
                    #   later in the game
                    elif g.gameResult == 17:
                        in_vec = g.toNN_vecs(every=False)[0]
                        stack[-1]['nn_inputs'].append(in_vec)
                        self.numLeaves += 1
                    else:
                        stack[-1]['nn_inputs'].append(None)
                        self.numLeaves += 1

            else:   # otherwise hop down one node
                processNode(self)


#   Descend one step of depth in the search after all moves for a node have been
#   explored. Pass reward down (minimax), along with alpha or beta as
#   appropriate.
def processNode(trav):
    node = trav.stack.pop()

    #   Adjust reward by adding NN evaluations, for whichever positions (if any)
    #   are at max depth and are unfinished games
    indices = [
        i for i in range(len(node['nn_inputs']))
        if node['nn_inputs'][i] is not None
    ]
    if len(indices) >= 1:
        if trav.fakeEvals:
            #   Just fake a reward of 0.1 for non-terminal positions
            for i in indices:
                node['rewards'][i] += 0.1
        else:
            #   Get the real NN evaluation(s)
            nn_vecs = tf.stack(
                [tf.reshape(node['nn_inputs'][i], [839]) for i in indices]
            )
            nn_out = trav.net(nn_vecs, training=False)[-1]
            
            nn_evals = trav.p['gamma_exec'] * nn_out
                
            for i in range(nn_evals.shape[0]):
                node['rewards'][indices[i]] += float(nn_evals[i])

    #   Pass reward down or set trav.baseR (whichever is applicable);
    #   update alpha or beta if applicable
    if node['game'].whiteToMove:
        index = np.argmax(node['rewards'])
        r = trav.p['gamma_exec'] * node['rewards'][index]
        this_line = node['move_names'][index]
        best_move = node['moves_done'][index]
        
        #   Update beta if necessary
        if len(trav.stack) > 0:
            trav.stack[-1]['beta'] = min(
                trav.stack[-1]['beta'], trav.stack[-1]['prev_reward'] + r
            )
    else:
        index = np.argmin(node['rewards'])
        r = trav.p['gamma_exec'] * node['rewards'][index]
        this_line = node['move_names'][index]
        best_move = node['moves_done'][index]

        #   Update alpha if necessary
        if len(trav.stack) > 0:
            trav.stack[-1]['alpha'] = max(
                trav.stack[-1]['alpha'], trav.stack[-1]['prev_reward'] + r
            )

    #   Pass reward and best move name down if applicable
    assert len(this_line) >= 1, len(this_line)
    if len(trav.stack) > 0:
        trav.stack[-1]['rewards'][-1] += r
        trav.stack[-1]['move_names'][-1] += this_line
    else:
        trav.baseR = r / trav.p['gamma_exec']
        trav.bestMove = best_move
        trav.bestLine = this_line
