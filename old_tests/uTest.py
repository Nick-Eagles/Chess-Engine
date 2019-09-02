import Move
import board_helper

import numpy as np
import copy

def isAmb(self, board):
    piece = board[self.startSq[0]][self.startSq[1]]

    if piece == 1:
        #   While technically not ambiguous, all pawn captures include file
        if self.startSq[0] != self.endSq[0]:
            return (False, True)
        
        return (False, False)
        
    elif piece == 2:
        #   From the move's end square, move like a knight and see if you end up on a square
        #   with a knight more than once (since the start square is not ignored)
        horizontals = [-2, -2, -1, -1, 1, 1, 2, 2]
        verticals = [-1, 1, -2, 2, -2, 2, -1, 1]
        xHits = 0   # number of knights on the same file as the moving knight
        yHits = 0   #   " same rank
        hits = 0
        for x,y in zip(horizontals, verticals):
            if board_helper.inBounds(self.endSq, (x,y)) and board[self.endSq[0] + x][self.endSq[1] + y] == 2:
                xHits += (self.endSq[0] + x == self.startSq[0])
                yHits += (self.endSq[1] + y == self.startSq[1])
                hits += 1
        #   If there are other knights but none that match on rank or file, specify file by default.
        #   Otherwise use normal rules for ambiguity
        if hits > 1 and xHits + yHits == 2:
            return (False, True)
        else:
            return (xHits > 1, yHits > 1)

    elif piece != 6:
        #return (False, False) # until I fix what's going on
        #   Use the less efficient, but convenient workaround for the remaining piece types:
        #   Pretend the end square holds the black king. Remove pieces of the target type until
        #   inCheck == False, and keep track of the ranks and files of the offending pieces.
        newBoard = copy.deepcopy(board)
        hits = 0
        xHits = 0
        yHits = 0

        #   Set original square to knight because we want a solid piece outside the
        #   set we're testing for, and unable to produce check in the same way
        newBoard[self.startSq[0]][self.startSq[1]] = 2
            
        samePieces = []
        for f in range(8):
            for r in range(8):
                p = newBoard[f][r]
                if p == 12:
                    newBoard[f][r] = 0
                elif p == piece:
                    #   We'll invert the board soon
                    samePieces.append([f, r])
                elif p < 6 and p != 0:
                    #   Make irrelevant white pieces black knights; this removes the possibility
                    #   of irrelevant "checks" while preserving ability to block relevant "checks"
                    newBoard[f][r] = 8

        newBoard[self.endSq[0]][self.endSq[1]] = 12
            
        #   Make sure there are candidate matches
        if len(samePieces) == 0:
            return (False, False)

        invBoard = board_helper.invert(newBoard)
        while board_helper.inCheck(invBoard):
            sq = samePieces.pop()
            xHit = sq[0] == self.startSq[0]
            yHit = sq[1] == self.startSq[1]
            hits += 1
            xHits += xHit
            yHits += yHit
            invBoard[7-sq[0]][7-sq[1]] = 0

        if hits > 0 and xHits + yHits == 0:
            return (False, True)
        else:
            return (xHits > 0, yHits > 0)

    return (False, False)    # (since piece = 6 and there must be one king)


board = np.zeros((8,8), dtype=np.int8)
board = [[4,0,0,0,1,0,0,0],[2,0,0,0,0,7,0,5],[0,0,1,0,0,9,7,0],
         [0,0,0,0,0,0,12,0],[6,3,0,0,1,7,0,11],[0,4,0,0,0,0,0,5],
         [2,0,0,0,0,0,0,0],[0,0,1,0,0,0,1,10]]
#board[1][7] = 5
#board[5][7] = 5
#board[3][6] = 12
#board[4][1] = 6
#board[4][7] = 11
#board = board.tolist()

ambMove = Move.Move((1,7), (4,7), 5)
#print(board)
#print(ambMove.getMoveName(board, True))
print(isAmb(ambMove, board))
