import board_helper

#   Encapsulates information that comprises a chess move and provides a function
#   to return the formal move name (ex. Qxe7). Does not support "+" or "#".
class Move:
    def __init__(self, startSq, endSq, endPiece):
        assert endPiece != 0, "Tried to create move with self-deleting piece"
        assert (startSq[0] != endSq[0] or startSq[1] != endSq[1]), "Tried to create a move that does nothing"
        
        #   These are tuples of integers in [0,7]x[0,7]
        self.startSq = startSq
        self.endSq = endSq

        #   If piece changes (=> pawn promotion)
        self.endPiece = endPiece

    def copy(self):
        startSq = (self.startSq[0], self.startSq[1])
        endSq = (self.endSq[0], self.endSq[1])

        return Move(startSq, endSq, self.endPiece)

    def invert(self):
        startSq = (7 - self.startSq[0], 7 - self.startSq[1])
        endSq = (7 - self.endSq[0], 7 - self.endSq[1])
        endPiece = -1 * self.endPiece

        return Move(startSq, endSq, endPiece)

    def isCapture(self, board):
        #   The end square is occupied
        if board[self.endSq[0]][self.endSq[1]] != 0:
            return True
        
        #   The capture was en passant
        if abs(board[self.startSq[0]][self.startSq[1]]) == 1 and board[self.endSq[0]][self.endSq[1]] == 0 and self.startSq[0] != self.endSq[0]:
            return True
        return False

    def equals(self, move):
        return self.startSq == move.startSq and self.endSq == move.endSq and self.endPiece == move.endPiece

    def toString(self):
        text = "self.startSq: " + str(self.startSq) + "\n"
        text += "self.endSq: " + str(self.endSq) + "\n"
        text += "self.endPiece: " + str(self.endPiece)
        return text
    
    #   Prints the move name in chess standard notation.
    #   NOTE: the move is assumed to be legitimate; function does not check
    #   legality of move.
    def getMoveName(self, board):
        piece = abs(board[self.startSq[0]][self.startSq[1]])
        assert piece != 0, "Tried to name a move that started on an empty square:\n" + self.toString()
        assert piece
            
        #   Piece letter
        if piece == 1:
            move = ''
        elif piece == 2:
            move = 'N'
        elif piece == 3:
            move = 'B'
        elif piece == 4:
            move = 'R'
        elif piece == 5:
            move = 'Q'
        elif piece == 6:
            if self.endSq[0] - self.startSq[0] == 2:
                return "O-O"
            elif self.endSq[0] - self.startSq[0] == -2:
                return "O-O-O"
            else:
                move = 'K'
        
        #   Deal with potential ambiguity
        amb = self.isAmbiguous(board)
        if amb[1]:
            move += chr(self.startSq[0]+97)
        if amb[0]:
            move += str(self.startSq[1]+1)

        #   Deal with possibility of capture
        if self.isCapture(board):
            move += 'x'

        #   End square
        move += chr(self.endSq[0]+97) + str(self.endSq[1]+1)

        #   Special pawn stuff
        if piece == 1:
            #   End sqaure is empty, meaning an en passant occurred
            if self.startSq[0] != self.endSq[0] and board[self.endSq[0]][self.endSq[1]] == 0:
                move = chr(self.startSq[0]+97) + 'x' + chr(self.endSq[0]+97) + str(self.endSq[1]+1)
                move += ' e.p.'
                
            #   Handles promotion
            elif self.endSq[1] == 0 or self.endSq[1] == 7:
                assert abs(self.endPiece) >= 2 and abs(self.endPiece) <= 5, "Tried to promote to piece " + str(self.endPiece)
                if abs(self.endPiece) == 2:
                    move += '=N'
                elif abs(self.endPiece) == 3:
                    move += '=B'
                elif abs(self.endPiece) == 4:
                    move += '=R'
                else:
                    move += '=Q'

        return move

    #   Check if more than one piece would be able to do the move name if the starting file or
    #   rank were not clarified. Returns a 2-tuple of booleans representing if file and/or rank
    #   is ambiguous, in that order.
    def isAmbiguous(self, board):
        piece = board[self.startSq[0]][self.startSq[1]]
        coeff = 2 * (piece > 0) - 1

        if abs(piece) == 1:
            #   While technically not ambiguous, all pawn captures include file
            if self.startSq[0] != self.endSq[0]:
                return (False, True)
        
            return (False, False)
        
        elif abs(piece) == 2:
            #   From the move's end square, move like a knight and see if you end up on a square
            #   with a knight more than once (since the start square is not ignored)
            horizontals = [-2, -2, -1, -1, 1, 1, 2, 2]
            verticals = [-1, 1, -2, 2, -2, 2, -1, 1]
            xHits = 0   # number of knights on the same file as the moving knight
            yHits = 0   #   " same rank
            hits = 0
            for x,y in zip(horizontals, verticals):
                if board_helper.inBounds(self.endSq, (x,y)) and board[self.endSq[0] + x][self.endSq[1] + y] == piece:
                    xHits += (self.endSq[0] + x == self.startSq[0])
                    yHits += (self.endSq[1] + y == self.startSq[1])
                    hits += 1
            #   If there are other knights but none that match on rank or file, specify file by default.
            #   Otherwise use normal rules for ambiguity
            if hits > 1 and xHits + yHits == 2:
                return (False, True)
            else:
                return (xHits > 1, yHits > 1)

        elif abs(piece) != 6:
            #   Use the less efficient, but convenient workaround for the remaining piece types:
            #   Pretend the end square holds the opposite king. Remove pieces of the target type until
            #   inCheck == False, and keep track of the ranks and files of the offending pieces.
            newBoard = [x.copy() for x in board]
            hits = 0
            xHits = 0
            yHits = 0

            #   Set original square to knight because we want a solid piece outside the
            #   set we're testing for, and unable to produce check in the same way
            newBoard[self.startSq[0]][self.startSq[1]] = 2*coeff
            
            samePieces = []
            for f in range(8):
                for r in range(8):
                    p = newBoard[f][r]
                    if p == -6*coeff:
                        newBoard[f][r] = 0
                    elif p == piece:
                        #   We'll invert the board soon
                        samePieces.append([f, r])
                    elif p != 0:
                        #   Make irrelevant white pieces black knights; this removes the possibility
                        #   of irrelevant "checks" while preserving ability to block relevant "checks"
                        newBoard[f][r] = -2*coeff

            newBoard[self.endSq[0]][self.endSq[1]] = -6*coeff
            
            #   Make sure there are candidate matches
            if len(samePieces) == 0:
                return (False, False)

            if piece > 0:
                newBoard = board_helper.invert(newBoard)
            while board_helper.inCheck(newBoard):
                sq = samePieces.pop()
                xHit = sq[0] == self.startSq[0]
                yHit = sq[1] == self.startSq[1]
                hits += 1
                xHits += xHit
                yHits += yHit
                if piece > 0:
                    newBoard[7-sq[0]][7-sq[1]] = 0
                else:
                    newBoard[sq[0]][sq[1]] = 0

            if hits > 0 and xHits + yHits == 0:
                return (False, True)
            else:
                return (xHits > 0, yHits > 0)

        return (False, False)    # (since piece = 6 and there must be one king)
