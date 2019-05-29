import random
import numpy as np
    
def moveUp(in_board):
    scores = np.zeros(4)

    out_board = np.empty((4,4))
    for i in range(4):
        tiles = in_board[i][~np.isnan(in_board[i])]
        out_board[i] = np.array([np.NaN]*4)
        out_cnt = 0
        in_cnt = 0
        while in_cnt < tiles.shape[0]-1:
            if tiles[in_cnt] == tiles[in_cnt+1]:
                out_board[i][out_cnt] = tiles[in_cnt]*2
                scores[i] += out_board[i][out_cnt]
                out_cnt += 1
                in_cnt += 2
            else:
                out_board[i][out_cnt] = tiles[in_cnt]
                in_cnt += 1
                out_cnt += 1
            
        if in_cnt == tiles.shape[0]-1:
            out_board[i][out_cnt] = tiles[in_cnt]
            out_cnt += 1 

    return out_board, scores.sum()

class Game(object):
    def __init__(self):
        self.board = np.empty((4,4))
        self.board[:] = np.NaN
        self.score = 0
        self.over = False
        a = random.choice([0,1,2,3])
        b = random.choice([0,1,2,3])
        c = random.choice([0,1,2,3])
        d = random.choice([0,1,2,3])
        while a == c and b == d:
            d = random.choice([0,1,2,3])
        self.board[a][b] = 2
        self.board[c][d] = 2
    
    def show(self):
        print('Score: %d' % self.score)
        for i in range(4):
            print('\n_________________________')
            print('\n|', end='')
            for j in range(4):
                if np.isfinite(self.board[i][j]):
                    print("%4d " % self.board[i][j], end='')
                else:
                    print("     ", end='')
                print('|', end='')
        print('\n_________________________')
        if self.over:
            print('\n      GAME OVER       ')
        
    def getEmptyCells(self):
        empty = np.transpose(np.nonzero(np.isnan(self.board)))
        return empty

    def move(self, direction):
        if self.over:
            return

        new_board, score_change = moveUp(np.rot90(self.board, k=direction))
        new_board = np.rot90(new_board, k=-direction)
        
        if np.allclose(new_board, self.board, equal_nan=True):
            return
        
        self.score += score_change
        self.board = new_board
        x,y = random.choice(self.getEmptyCells())
        if random.random() < 0.9:
            self.board[x][y] = 2
        else:
            self.board[x][y] = 4
            
        if np.count_nonzero(np.isnan(self.board)) == 0:
            if np.allclose(moveUp(self.board)[0], self.board) and \
             np.allclose(np.rot90(moveUp(np.rot90(self.board, 1))[0],-1), self.board) and \
             np.allclose(np.rot90(moveUp(np.rot90(self.board, 2))[0],-2), self.board) and \
             np.allclose(np.rot90(moveUp(np.rot90(self.board, -1))[0],1), self.board):
                self.over = True