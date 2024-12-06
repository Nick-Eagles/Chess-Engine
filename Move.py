import board_helper
import misc

#   Encapsulates information that comprises a chess move and provides a function
#   to return the formal move name (ex. Qxe7). Does not support "+" or "#".
class Move:
    def __init__(self, startSq, endSq, endPiece):
        assert endPiece != 0, "Tried to create move with self-deleting piece"
        assert (startSq[0] != endSq[0] or startSq[1] != endSq[1]), \
               "Tried to create a move that does nothing"
        assert endPiece >= -6 and endPiece <= 6, \
               "Tried to create an invalid piece type: " + str(endPiece)
        
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
        return self.startSq == move.startSq and self.endSq == move.endSq and \
               self.endPiece == move.endPiece

    def toString(self):
        text = "self.startSq: " + str(self.startSq) + "\n"
        text += "self.endSq: " + str(self.endSq) + "\n"
        text += "self.endPiece: " + str(self.endPiece)
        return text
    
    #   Prints the move name in algebraic notation. NOTE: the move is assumed to
    #    be legitimate; function does not check legality of move.
    def getMoveName(self, board):
        piece = abs(board[self.startSq[0]][self.startSq[1]])
        assert piece != 0, \
               "Tried to name a move that started on an empty square:\n" + \
               self.toString()
            
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

        #   Handles promotion
        if piece == 1 and self.endSq[1] == 0 or self.endSq[1] == 7:
            assert abs(self.endPiece) >= 2 and abs(self.endPiece) <= 5, \
                    "Tried to promote to piece " + str(self.endPiece)
            if abs(self.endPiece) == 2:
                move += '=N'
            elif abs(self.endPiece) == 3:
                move += '=B'
            elif abs(self.endPiece) == 4:
                move += '=R'
            else:
                move += '=Q'

        return move

    #   Check if more than one piece would be able to do the move name if the
    #   starting file or rank were not clarified. Returns a 2-tuple of booleans
    #   representing if file and/or rank is ambiguous, in that order.
    def isAmbiguous(self, board):
        piece = board[self.startSq[0]][self.startSq[1]]
        coeff = 2 * (piece > 0) - 1

        if abs(piece) == 1:
            #   While technically not ambiguous, all pawn captures include file
            if self.startSq[0] != self.endSq[0]:
                return (False, True)
        
            return (False, False)
        
        elif abs(piece) == 2:
            #   From the move's end square, move like a knight and see if you
            #   end up on a square with a knight more than once (since the start
            #   square is not ignored)
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
            #   If there are other knights but none that match on rank or file,
            #   specify file by default. Otherwise use normal rules for
            #   ambiguity
            if hits > 1 and xHits + yHits == 2:
                return (False, True)
            else:
                return (xHits > 1, yHits > 1)

        elif abs(piece) != 6:
            #   Describe the "one square" movements of each piece type by
            #   (file change, rank change)
            bishop_moves = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            rook_moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            all_moves = [bishop_moves, rook_moves, bishop_moves + rook_moves]

            these_moves = all_moves[misc.match(abs(piece), [3, 4, 5])]

            #   This will be a list of (file, rank) pairs for pieces of the
            #   same type
            key_squares = []

            for motion in these_moves:
                #   Do we know yet if the piece of interest is along the current
                #   "line of motion"?
                decided = False
                i = 1

                #   Move along a line until we encounter a piece. Record the
                #   file and rank if it matches the piece type
                while not decided and board_helper.inBounds(self.endSq, (i*motion[0], i*motion[1])):
                    current_piece = board[self.endSq[0] + i*motion[0]][self.endSq[1] + i*motion[1]]
                    if current_piece == piece:
                        #   We found a piece of the same type
                        key_squares.append(
                            (self.endSq[0] + i*motion[0],
                             self.endSq[1] + i*motion[1])
                        )
                        decided = True
                    elif current_piece != 0:
                        #   The first piece along the "line" is different than
                        #   the type we are looking for
                        decided = True
                    else:
                        i += 1
            
            file_matches = sum([x[0] == self.startSq[0]
                                for x in key_squares]) > 1
            rank_matches = sum([x[1] == self.startSq[1]
                                for x in key_squares]) > 1
            if len(key_squares) > 1:
                #   More than one piece can move to the same square, but these
                #   pieces are on different files and ranks. In this case,
                #   specify the file
                if not file_matches and not rank_matches:
                    return (False, True)
                else:
                    return (file_matches, rank_matches)
            else:
                return (False, False)
        else:
            #   King moves are never ambiguous
            return (False, False)
