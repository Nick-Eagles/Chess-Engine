import numpy as np
import datetime
import tensorflow as tf
import re

import Move
import board_helper

class Game:
    def __init__(self, quiet=True):
        self.board = [
            [4, 1, 0, 0, 0, 0, -1, -4], [2, 1, 0, 0, 0, 0, -1, -2],
            [3, 1, 0, 0, 0, 0, -1, -3], [5, 1, 0, 0, 0, 0, -1, -5],
            [6, 1, 0, 0, 0, 0, -1, -6], [3, 1, 0, 0, 0, 0, -1, -3],
            [2, 1, 0, 0, 0, 0, -1, -2], [4, 1, 0, 0, 0, 0, -1, -4]
        ]
        self.invBoard = [
            [4, 1, 0, 0, 0, 0, -1, -4], [2, 1, 0, 0, 0, 0, -1, -2],
            [3, 1, 0, 0, 0, 0, -1, -3], [6, 1, 0, 0, 0, 0, -1, -6],
            [5, 1, 0, 0, 0, 0, -1, -5], [3, 1, 0, 0, 0, 0, -1, -3],
            [2, 1, 0, 0, 0, 0, -1, -2], [4, 1, 0, 0, 0, 0, -1, -4]
        ]
        
        #   Amount of the pieces on the board: P, N, light B, dark B, R, Q
        self.wPieces = [8, 2, 1, 1, 2, 1]
        self.bPieces = [8, 2, 1, 1, 2, 1]

        #   These only become false when castling has occurred or a rook
        #   or king has moved, making castling illegal *for the game*.
        self.canW_K_Castle = True
        self.canW_Q_Castle = True
        self.canB_K_Castle = True
        self.canB_Q_Castle = True

        self.wValue = 43
        self.bValue = 43

        self.whiteToMove = True
        self.enPassant = False
        self.currentlyCheck = False
        self.quiet = quiet

        self.movesSinceAction = 0
        self.gameResult = 17    #   Literally a random number, but will become +/-1 or 0
        self.gameResultStr = "Not a terminal position"
        self.moveNum = 1
        self.annotation = ""
        self.lastMove = Move.Move((0, 6), (0, 7), -1) # No particular meaning, simply initialized

    def copy(self):
        g = Game()

        g.board = [x.copy() for x in self.board]
        g.invBoard = [x.copy() for x in self.invBoard]
        g.wPieces = self.wPieces.copy()
        g.bPieces = self.bPieces.copy()
        g.canW_K_Castle = self.canW_K_Castle
        g.canW_Q_Castle = self.canW_Q_Castle
        g.canB_K_Castle = self.canB_K_Castle
        g.canB_Q_Castle = self.canB_Q_Castle
        g.wValue = self.wValue
        g.bValue = self.bValue
        g.whiteToMove = self.whiteToMove
        g.enPassant = self.enPassant
        g.currentlyCheck = self.currentlyCheck
        g.quiet = self.quiet
        g.movesSinceAction = self.movesSinceAction
        g.gameResult = self.gameResult
        g.gameResultStr = self.gameResultStr
        g.moveNum = self.moveNum
        g.annotation = self.annotation
        g.lastMove = self.lastMove

        return g

    def updateValues(self):
        vals = [1, 3, 3, 3, 5, 9]

        self.wValue = sum([vals[i] * self.wPieces[i] for i in range(6)]) + 4
        self.bValue = sum([vals[i] * self.bPieces[i] for i in range(6)]) + 4

    #   Return the (absolute reward for doing move "move" (positive means to the
    #   benefit of white), NN input vector for the resulting position) as a
    #   tuple
    def getReward(self, move, mateRew, simple=False, copy=True):
        if copy:
            g = self.copy()
            g.quiet = True
        else:
            g = self

        #   Save original material values, since g may be self
        old_b_value = g.bValue
        old_w_value = g.wValue
        
        g.doMove(move)

        if simple:
            NN_vecs = None
        else:
            NN_vecs = g.toNN_vecs()
            
        if abs(g.gameResult) == 1:
            r = g.gameResult * mateRew
        elif g.gameResult == 0:
            r = np.log(old_b_value / old_w_value)
        else:
            r = np.log(g.wValue * old_b_value / (old_w_value * g.bValue))

        return (r, NN_vecs)

    def printBoard(self):
        print("board:  -----")
        for rank in range(8):
            for file in range(8):
                piece = self.board[file][7-rank]
                print(piece, end = ' ')
            print()
        print("-------------")

    #   Converts the board into a standardized input vector for the NN:
    #   for each square, the input vector has 12 entries so that at most one
    #   entry has a value of 1 (the remaining are zero), corresponding to
    #   the piece occupying that square. The remaining pieces of information
    #   necessary to specify a unique board are appended to the vector:
    #   game.enPassant, castling permissions, and moves since a pawn move or
    #   capture. All entries are again 0/1 except game.movesSinceAction (which
    #   is still in [0, 1))
    def toNN_vecs_old(self):
        netInput = np.zeros((774,))

        c = 0
        if self.whiteToMove:
            for file in range(8):
                for rank in range(8):
                    piece = self.board[rank][file]
                    if piece > 0:
                        netInput[c + piece + 5] = 1
                    elif piece < 0:
                        netInput[c + piece + 6] = 1
                    c += 12
        else:
            for file in range(8):
                for rank in range(8):
                    piece = -1 * self.board[file][7-rank]
                    if piece > 0:
                        netInput[c + piece + 5] = 1
                    elif piece < 0:
                        netInput[c + piece + 6] = 1
                    c += 12

        netInput[768] = int(self.enPassant)
        if self.whiteToMove:
            netInput[769] = int(self.canW_K_Castle)
            netInput[770] = int(self.canW_Q_Castle)
            netInput[771] = int(self.canB_K_Castle)
            netInput[772] = int(self.canB_Q_Castle)
        else:
            netInput[769] = int(self.canB_K_Castle)
            netInput[770] = int(self.canB_Q_Castle)
            netInput[771] = int(self.canW_K_Castle)
            netInput[772] = int(self.canW_Q_Castle)
            
        #   Normalize so it fits in [0, ~1)
        netInput[773] = self.movesSinceAction / 50

        return tf.constant(netInput, shape=(1,774), dtype=tf.float32)
    
    #   Convert a Game to an input tensor for a convolutional neural network
    def toNN_vecs(self):
        net_input = np.zeros((1, 8, 8, 15), dtype = np.float32)

        #   Fill in the piece occupancy channels of the input
        for file in range(8):
            for rank in range(8):
                #   Select the piece at this rank and file. Invert the board by
                #   color and rank if black
                if self.whiteToMove:
                    piece = self.board[file][rank]
                else:
                    piece = -1 * self.board[file][7-rank]
                
                #   Ensure piece has values in {0, 1, ..., 11}. Each piece is a
                #   different channel
                if piece > 0:
                    net_input[0, file, rank, piece + 5] = 1
                elif piece < 0:
                    net_input[0, file, rank, piece + 6] = 1

        #   Fill in the castling channel of the input. Here a 1 is placed on
        #   every square where either king can land after legally castling
        if (self.whiteToMove and self.canW_K_Castle) or (not self.whiteToMove and self.canB_K_Castle):
            net_input[0, 6, 0, 12] = 1
        if (self.whiteToMove and self.canW_Q_Castle) or (not self.whiteToMove and self.canB_Q_Castle):
            net_input[0, 2, 0, 12] = 1
        if (self.whiteToMove and self.canB_K_Castle) or (not self.whiteToMove and self.canW_K_Castle):
            net_input[0, 6, 7, 12] = 1
        if (self.whiteToMove and self.canB_Q_Castle) or (not self.whiteToMove and self.canW_Q_Castle):
            net_input[0, 2, 7, 12] = 1
        
        #   Fill in the en-passant channel of the input. Place a 1 where there's
        #   a pawn that can be captured en passant
        if self.enPassant:
            if self.whiteToMove:
                net_input[0, self.lastMove.endSq[0], self.lastMove.endSq[1], 13] = 1
            else:
                net_input[0, self.lastMove.endSq[0], 7 - self.lastMove.endSq[1], 13] = 1
        
        #   Normalize so it fits in [0, ~1)
        net_input[0, :, :, 14] = self.movesSinceAction / 50

        return tf.constant(net_input, shape=(1, 8, 8, 15), dtype=tf.float32)

    #   Returns a tuple: the first entry is the game result; the second is a string
    #   describing details (such as "Draw by stalemate")
    def updateResult(self):
        coeff = 1 - 2 * self.whiteToMove

        ########################################################
        #   See if the position is checkmate
        ########################################################

        #   Update variables 'currentlyCheck' and inverted board
        if self.whiteToMove:
            self.currentlyCheck = board_helper.inCheck(self.board)
        else:
            self.invBoard = board_helper.invert(self.board)
            self.currentlyCheck = board_helper.inCheck(self.invBoard)

        any_moves = board_helper.anyLegalMoves(self)

        if self.currentlyCheck and not any_moves:
            return (coeff, "Checkmate")

        #   Rule out any draws
        if self.moveNum < 10:
            return (17, "Not a terminal position")

        ########################################################
        #   Stalemate
        ########################################################

        if not self.currentlyCheck and not any_moves:
            return (0, "Draw by stalemate.")

        ########################################################
        #   50 MOVE RULE
        ########################################################
        
        if self.movesSinceAction >= 50:
            note = "Draw: no capture or pawn advance in 50 moves."
            return (0, note)

        ########################################################  
        #   Insufficient material
        ########################################################
        
        #   Checks for any pawns, rooks, or queens
        if sum([self.wPieces[i] + self.bPieces[i] for i in [0, 4, 5]]) == 0:  
            #   A bool representing if either side has both bishop colors (which would mean
            #   the game is not a draw)
            bothColors =(self.wPieces[2] > 0 and self.wPieces[3] > 0) or \
                        (self.bPieces[2] > 0 and self.bPieces[3] > 0)
            if not bothColors:
                #   A bool ultimately deciding whether both sides have only bishops
                #   of one color
                colorMatch = (self.wPieces[2] > 0) == (self.bPieces[2] > 0)
                numKnights = self.wPieces[1] + self.bPieces[1]
                
                #   Equivalent to checking if there are bishops on the board and
                #   they are all on the same color
                if colorMatch and (self.wPieces[2] + self.wPieces[3] > 0) or (self.wPieces[3] + self.wPieces[2] > 0):
                    if numKnights == 0:
                        note = "Draw by insufficient material."
                        return (0, note)
                elif colorMatch:    # in this case meaning no bishops are on the board
                    if numKnights < 2:
                        note = "Draw by insufficient material."
                        return (0, note)

        return (17, "Not a terminal position")

    #   Changes the board and relevant class attributes for any move; does not entirely
    #   verify legitimacy of move.
    def doMove(self, move):
        coeff = 2 * self.whiteToMove - 1
        assert move.endPiece != 0, "Piece was likely deleted.\n" + self.verbosePrint()
        assert coeff * move.endPiece > 0, "Tried to move opponent's piece.\n" + self.verbosePrint()
        assert self.wValue >= 4 and self.bValue >= 4, "See what went wrong, saving game to PGN:" + str(self.toPGN()) + self.verbosePrint()
        
        self.lastMove = move
        self.enPassant = False

        if self.whiteToMove:
            if not self.quiet:
                self.annotation += str(self.moveNum) + ". " + move.getMoveName(self) + " "
            piecesList = self.wPieces
            oppPiecesList = self.bPieces
        else:
            if not self.quiet:
                self.annotation += move.getMoveName(self) + "\n"
            piecesList = self.bPieces
            oppPiecesList = self.wPieces

        startPiece = self.board[move.startSq[0]][move.startSq[1]]
        pieceRemoved = self.board[move.endSq[0]][move.endSq[1]]
        assert startPiece != 0, "Tried to move a piece from an empty square: " + move.getMoveName(self) + self.verbosePrint()
        assert coeff*pieceRemoved <= 0, "Tried to capture own piece: " + move.getMoveName(self) + self.verbosePrint()
        assert abs(pieceRemoved) != 6, "About to capture opponent's king: " + move.getMoveName(self) + self.verbosePrint()       

        #   Sees if the move is a capture or pawn move
        if pieceRemoved != 0 or abs(startPiece) == 1:
            self.movesSinceAction = 0
        else:
            self.movesSinceAction += 0.5

        #   Identify when a pawn moves 2 squares and creates an
        #   en passant opportunity
        if abs(startPiece) == 1 and abs(move.endSq[1] - move.startSq[1]) == 2:
            #   Opposite color pawn on the left of where this pawn moved
            if move.startSq[0] > 0 and self.board[move.endSq[0]-1][move.endSq[1]] == -1*startPiece:
                self.enPassant = True
            #   To the right
            elif move.startSq[0] < 7 and self.board[move.endSq[0]+1][move.endSq[1]] == -1*startPiece:
                self.enPassant = True
            
        #   In case of promotion, properly update number of same color pieces
        #   of each type
        if startPiece != move.endPiece:
            assert coeff*move.endPiece >= 2 and coeff*move.endPiece <= 5, "Tried to promote to value " + move.endPiece + ".\n" + self.verbosePrint()
            piecesList[0] -= 1
            if abs(move.endPiece) == 3:
                if (move.endSq[0] - move.endSq[1]) % 2 == 0:
                    piecesList[3] += 1
                else:
                    piecesList[2] += 1
            elif abs(move.endPiece) >= 4:
                piecesList[coeff*move.endPiece] += 1
            else:   # == 2
                piecesList[1] += 1

        #   Update number of opposite color pieces of each type                  
        if abs(pieceRemoved) == 3:
            if (move.endSq[0] - move.endSq[1]) % 2 == 0:
                oppPiecesList[3] -= 1
            else:
                oppPiecesList[2] -= 1
        elif abs(pieceRemoved) >= 4:
            oppPiecesList[abs(pieceRemoved)] -= 1
        elif abs(pieceRemoved) == 1 or abs(pieceRemoved) == 2:
            oppPiecesList[abs(pieceRemoved)-1] -= 1
            
        #   Move rook to complete castling, if applicable
        backRank = 7*(not self.whiteToMove)
        if move.endSq[0] - move.startSq[0] == 2 and abs(startPiece) == 6:    # kingside castling
            self.board[5][backRank] = 4*coeff
            self.board[7][backRank] = 0
        elif move.endSq[0] - move.startSq[0] == -2 and abs(startPiece) == 6: # queenside castling
            self.board[3][backRank] = 4*coeff
            self.board[0][backRank] = 0

        #   Set castling permissions to false if king or rooks have moved
        if move.endPiece == 6:
            self.canW_K_Castle = False
            self.canW_Q_Castle = False
        elif move.endPiece == 4 and move.startSq[0] == 0 and move.startSq[1] == 0:
            self.canW_Q_Castle = False
        elif move.endPiece == 4 and move.startSq[0] == 7 and move.startSq[1] == 0:
            self.canW_K_Castle = False
        elif move.endPiece == -6:
            self.canB_K_Castle = False
            self.canB_Q_Castle = False
        elif move.endPiece == -4 and move.startSq[0] == 0 and move.startSq[1] == 7:
            self.canB_Q_Castle = False
        elif move.endPiece == -4 and move.startSq[0] == 7 and move.startSq[1] == 7:
            self.canB_K_Castle = False

        #   Also revoke permissions for opposite color if rook is captured
        if move.endSq == (0, 0):
            self.canW_Q_Castle = False
        elif move.endSq == (7, 0):
            self.canW_K_Castle = False
        elif move.endSq == (0, 7):
            self.canB_Q_Castle = False
        elif move.endSq == (7, 7):
            self.canB_K_Castle = False

        #   Modify board
        self.board[move.startSq[0]][move.startSq[1]] = 0                
        self.board[move.endSq[0]][move.endSq[1]] = move.endPiece

        #   Capture the pawn if en passant
        if pieceRemoved == 0 and abs(startPiece) == 1 and move.startSq[0] != move.endSq[0]:
            self.board[move.endSq[0]][move.endSq[1]-coeff] = 0
            oppPiecesList[0] -= 1

        if not self.whiteToMove:
            self.moveNum += 1
            
        if self.whiteToMove and not self.quiet and self.moveNum % 20 == 0:
            print("Done move", self.moveNum)           
        
        self.whiteToMove = not self.whiteToMove
        self.updateValues()

        #   See if resulting position (after moving) is a checkmate or draw
        self.gameResult, self.gameResultStr = self.updateResult()
        

    #   For debugging; prints info about the current game and returns the empty
    #   string, so that assert statements can easily include a call to this
    #   function
    def verbosePrint(self):
        coeff = 2 * self.whiteToMove - 1
        if self.whiteToMove:
            agent = "white"
        else:
            agent = "black"
            
        print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(" Info about the current game:")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("self.canW_K_Castle: ", self.canW_K_Castle)
        print("self.canW_Q_Castle: ", self.canW_Q_Castle)
        print("self.canB_K_Castle: ", self.canB_K_Castle)
        print("self.canB_Q_Castle: ", self.canB_Q_Castle)
        print("self.enPassant: ", self.enPassant)
        print("self.movesSinceAction: ", self.movesSinceAction)
        print("self.moveNum: ", self.moveNum)
        print(
            "self.gameResult (will be 17 if game isn't done): ",
            self.gameResult
        )
        print("self.wValue:", self.wValue)
        print("self.bValue:", self.bValue)
        print("self.annotation: ", self.annotation)
        
        check = (self.whiteToMove and not board_helper.inCheck(self.board)) or \
                (not self.whiteToMove and not board_helper.inCheck(board_helper.invert(self.board)))
        print(
            "It is " + agent + " to move and " + agent + " is" +
            (check * " not") + " in check."
        )
        self.printBoard()

        return ""

    def toPGN(self, filename='visualization/latest_game.pgn'):
        date = datetime.datetime.now()
        dateStr = '"' + str(date.year) + '.' + str(date.month) + '.' + str(date.day) + '"'
        if self.gameResult == -1:
            result = '0-1'
        elif self.gameResult == 0:
            result = '1/2-1/2'
        else:
            result = '1-0'
            
        txt = '[Event "Engine self-play"]\n[Site "Internal"]\n[Date ' + \
            dateStr + ']\n' + \
            '[Round "1"]\n[White "Sculps Engine"]\n[Black "Sculps Engine"]\n' +\
            '[Result "' + result + '"]\n\n' + self.annotation + result

        with open(filename, 'w') as pgn_file:
            pgn_file.write(txt)
    
    #   Return a FEN string representing the current position in the Game
    def toFEN(self):
        fen_str = ''

        #   First fill in the board piece of the FEN string
        rank = 7
        while rank >= 0:
            file = 0
            num_empty = 0
            while file < 8:
                piece = self.board[file][rank]

                if piece == 0:
                    num_empty += 1
                    if file == 7:
                        fen_str += str(num_empty)
                else:
                    if num_empty > 0:
                        fen_str += str(num_empty)
                        num_empty = 0
                    fen_str += 'kqrbnp0PNBRQK'[piece + 6]
                
                file += 1
                
            if rank != 0:
                fen_str += '/'
            
            rank -= 1
        
        #   Player to move
        if self.whiteToMove:
            fen_str += ' w '
        else:
            fen_str += ' b '
        
        #   Castling permissions
        if self.canW_K_Castle:
            fen_str += 'K'
        if self.canW_Q_Castle:
            fen_str += 'Q'
        if self.canB_K_Castle:
            fen_str += 'k'
        if self.canB_Q_Castle:
            fen_str += 'q'

        #   En passant square if applicable
        if self.enPassant:
            rank_adj = 2 * self.whiteToMove - 1
            fen_str += f' {"abcdefgh"[self.lastMove.endSq[0]]}{self.lastMove.endSq[1] + rank_adj} '
        else:
            fen_str += ' - '
        
        #   Half and full move counters
        fen_str += f'{int(2 * self.movesSinceAction)} {self.moveNum}'

        return fen_str

    #   Create a Game object from a FEN string. There are 3 limitations
    #   currently:
    #
    #   - game.annotation is impossible to correctly specify
    #   - game.lastMove is impossible to correctly specify
    #   - game.enPassant is theoretically possible to specify, but not supported
    #     right now. The only consequences are that en passant is not seen as a
    #     legal move when it might really be
    @classmethod
    def fromFEN(cls, fen_str):
        game = cls()
        
        #   Start with an empty board and later fill in
        game.board = [[0 for i in range(8)] for j in range(8)]
        game.wPieces = [0] * 6
        game.bPieces = [0] * 6

        #   Fill in the board
        rank = 7
        index = 0
        while rank >= 0:
            file = 0
            while file < 8:
                if re.search(r'[1-8]', fen_str[index]):
                    #   If a number, just skip over the empty files
                    file += int(fen_str[index])
                else:
                    #   Otherwise fill the current square with its piece
                    piece = 'kqrbnp0PNBRQK'.index(fen_str[index]) - 6
                    game.board[file][rank] = piece

                    #   Update number of each piece
                    if piece in [-1, -2]:
                        game.bPieces[abs(piece) - 1] += 1
                    elif piece == -3:
                        #   Account for light vs. dark-squared bishops
                        game.bPieces[2 + (file + rank + 1) % 2] += 1
                    elif piece in [-4, -5]:
                        game.bPieces[abs(piece)] += 1
                    elif piece in [1, 2]:
                        game.wPieces[piece - 1] += 1
                    elif piece == 3:
                        #   Account for light vs. dark-squared bishops
                        game.wPieces[2 + (file + rank + 1) % 2] += 1
                    elif piece in [4, 5]:
                        game.wPieces[piece] += 1
                    
                    file += 1
                index += 1
            rank -= 1
            index += 1 # to step over '/'
        assert fen_str[index] in 'wb', fen_str[index]

        game.invBoard = board_helper.invert(game.board)

        game.whiteToMove = bool("bw".index(fen_str.split(' ')[1]))

        #   Set castling permissions
        castling_str = fen_str.split(' ')[2]
        game.canW_K_Castle = 'K' in castling_str
        game.canW_Q_Castle = 'Q' in castling_str
        game.canB_K_Castle = 'k' in castling_str
        game.canB_Q_Castle = 'q' in castling_str

        game.updateValues()

        game.movesSinceAction = int(fen_str.split(' ')[4]) / 2
        game.moveNum = int(fen_str.split(' ')[5])

        #   Update game result and game.inCheck
        game.gameResult, game.gameResultStr = game.updateResult()

        return game
