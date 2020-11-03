        #   Reward for best line found so far
        best_line = 2 * p['mateReward']
        if self.game.whiteToMove:
            best_line *= -1
                            #   Determine if the current line could even possibly be
                            #   better than the best so far. If not, prune the tree.
                            #if len(stack) % 2 == 1:
                                #if self.game.whiteToMove:
                                    #best_r = stack[-1][3] + p['gamma_exec'] * (r + MAX_R)
                                #else:
                                    #best_r = stack[-1][3] + p['gamma_exec'] * (r - MAX_R)
                            #else:
                                #if self.game.whiteToMove:
                                    #best_r = max(stack[-1][3] + p['gamma_exec'] * (r + p['gamma_exec']**2 * MAX_R),
                                                       #np.log(g.wValue / g.bValue))
                                #else:
                                    #best_r = min(stack[-1][3] + p['gamma_exec'] * (r - p['gamma_exec']**2 * MAX_R),
                                                       #np.log(g.bValue / g.wValue))

                            #   Could this line be better than the best so far?
                            #promising = (best_r > best_line and self.game.whiteToMove) or \
                                        #(best_r < best_line and not self.game.whiteToMove)
                            
                            if promising:    
                                moves, fullMovesLen = self.policy(self.net, g, p)
                                stack.append([moves, [], g, stack[-1][3] + p['gamma_exec'] * r, stack[-1][4], stack[-1][5]])
                                self.nodeHops += 1
                            else:
                                self.pruneCuts += 1
