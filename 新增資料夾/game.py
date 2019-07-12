import numpy as np
import tkinter

class Board(object):
    """
    board for the game
    """
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 15))
        self.height = int(kwargs.get('height', 15))
        self.states = {} # board states, key:move as location on the board, value:player as pieces type
        self.n_in_row = int(kwargs.get('n_in_row', 5)) # need how many pieces in a row to win
        self.players = [1, 2] # player1 and player2
        
    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not less than %d' % self.n_in_row)
        self.current_player = self.players[start_player]  # start player        
        self.availables = list(range(self.width * self.height)) # available moves 
        self.states = {} # board states, key:move as location on the board, value:player as pieces type
        self.last_move = -1

    def move_to_location(self, move):
        """       
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move  // self.width
        w = move  %  self.width
        return [h, w]

    def location_to_move(self, location):
        if(len(location) != 2):
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if(move not in range(self.width * self.height)):
            return -1
        return move

    def current_state(self): 
        """return the board state from the perspective of the current player
        shape: 4*width*height"""
        
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]                           
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0   
            square_state[2][self.last_move //self.width, self.last_move % self.height] = 1.0 # last move indication   
        if len(self.states)%2 == 0:
            square_state[3][:,:] = 1.0

        return square_state[:,::-1,:]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1] 
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        h = self.last_move // width
        w = self.last_move % width
        player = states[self.last_move]

        judge = 0
        for i in range(w - n + 1, w + n): #上至下
            if i < 0 or i >= width:
                continue
            judge = judge + 1 if(states.get(i + h * width, -1) == player) else 0
            if judge == 5:
                return True, player
        judge = 0
        for i in range(h - n + 1, h + n): #左至右
            if i < 0 or i >= height:
                continue
            judge = judge + 1 if(states.get(i * width + w, -1) == player) else 0
            if judge == 5:
                return True, player
            
        judge = 0
        i = h - n
        j = w + n
        for count in range(9): #左下到右上
            i += 1
            j -= 1
            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            judge = judge + 1 if(states.get(i * width + j, -1) == player) else 0
            if judge == 5:
                return True, player

        judge = 0
        i = h - n
        j = w - n
        for count in range(9): #左上到右下
            i += 1
            j += 1
            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            judge = judge + 1 if(states.get(i * width + j, -1) == player) else 0
            if judge == 5:
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):#            
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

class Point:
    def __init__(self, x, y):
        self.x = x;
        self.y = y;
        self.pixel_x = 30 + 30 * self.x
        self.pixel_y = 30 + 30 * self.y

class Game(object):
    """
    game server
    """
    def __init__(self, board, **kwargs):
        self.board = board
        self.top_of_cv = 0
    
    def click1(self, event): #click1 because keyword repetition
        current_player = self.board.get_current_player()
        if self.players[current_player].getStr() == "Human":
            i = (event.x) // 30
            j = (event.y) // 30
            ri = (event.x) % 30
            rj = (event.y) % 30
            i = i-1 if ri<15 else i
            j = j-1 if rj<15 else j
            move = self.board.location_to_move((i, j))
            if move in self.board.availables:
                if current_player==self.start_player.get_player_ind():
                    ban = self.forbidden(i,j,current_player)
                    if ban == 1:
                        print("長連禁手!!!")
                    elif ban == 2:
                        print("win")
                    elif ban == 3:
                        print("四四禁手")
                    elif ban == 4:
                        print("三三禁手")
                    if ban == 1 or ban == 3 or ban == 4:
                        return
                    if not self.top_of_cv == 0:
                        self.cv.delete(self.top_of_cv)
                    self.cv.create_oval(self.chess_board_points[i][j].pixel_x-10, self.chess_board_points[i][j].pixel_y-10, self.chess_board_points[i][j].pixel_x+10, self.chess_board_points[i][j].pixel_y+10, fill='black') 
                else:
                    if not self.top_of_cv == 0:
                        self.cv.delete(self.top_of_cv)
                    self.cv.create_oval(self.chess_board_points[i][j].pixel_x-10, self.chess_board_points[i][j].pixel_y-10, self.chess_board_points[i][j].pixel_x+10, self.chess_board_points[i][j].pixel_y+10, fill='white')
                self.top_of_cv = self.cv.create_oval(self.chess_board_points[i][j].pixel_x-10, self.chess_board_points[i][j].pixel_y-10, self.chess_board_points[i][j].pixel_x+10, self.chess_board_points[i][j].pixel_y+10, outline='red', width=3)
                self.board.do_move(move)
                
    def forbidden(self, h, w, player):
        height = self.board.height
        width = self.board.width
        states = self.board.states
        n = self.board.n_in_row
        restricted = [0,0] # 33, 44
        buffer = ["33333333333" for j in range(4)] # 3: 棋盤外
        
        for i in range(w - n, w + n + 1): #上至下
            if i < 0 or i >= width:
                continue
            buffer[0] = buffer[0][:i-w+5] + str(states.get(i + h * width, 0)) + buffer[0][i-w+6:]
            buffer[0] = buffer[0][:5] + str(player) + buffer[0][6:]
            
        for i in range(h - n, h + n + 1): #左至右
            if i < 0 or i >= height:
                continue
            buffer[1] = buffer[1][:i-h+5] + str(states.get(i * width + w, 0)) + buffer[1][i-h+6:]
            buffer[1] = buffer[1][:5] + str(player) + buffer[1][6:]

        i = h - n - 1
        j = w + n + 1
        for count in range(11): #左下至右上
            i += 1
            j -= 1
            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            buffer[2] = buffer[2][:count] + str(states.get(i * width + j, 0)) + buffer[2][count+1:]
            buffer[2] = buffer[2][:5] + str(player) + buffer[2][6:]
            
        i = h - n - 1
        j = w - n - 1
        for count in range(11): #左上至右下
            i += 1
            j += 1
            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            buffer[3] = buffer[3][:count] + str(states.get(i * width + j, 0)) + buffer[3][count+1:]
            buffer[3] = buffer[3][:5] + str(player) + buffer[3][6:]
            
        """print()
        print(buffer[0])
        print(buffer[1])
        print(buffer[2])
        print(buffer[3])
        print()"""

        for i in range(4):
            judge = 0
            for j in range(11):
                judge = judge+1 if(buffer[i][j]==str(player)) else 0
                if judge == 5:
                    return 1 if (j<10 and buffer[i][j+1]==str(player)) else 2 # 1長連禁手  2 win

        for i in range(4):
            restricted[1] = restricted[1]+1 if(buffer[i].find("011110")!=-1 or buffer[i].find("011112")!=-1 or buffer[i].find("211110")!=-1 or buffer[i].find("011113")!=-1 or buffer[i].find("311110")!=-1) else restricted[1]
            if buffer[i].find("10111") != -1:
                restricted[1] = restricted[1] if(buffer[i].find("1101110")!=-1 or buffer[i].find("0101111")!=-1 or buffer[i].find("1101111")!=-1) else restricted[1]+1        
            if buffer[i].find("11011") != -1:
                if(not (buffer[i].find("1110110")!=-1 or buffer[i].find("0110111")!=-1 or buffer[i].find("1110111")!=-1)):
                    restricted[1] = restricted[1]+1
                    #print(i, buffer[i].find("11011"))
                    if(not (buffer[i].find("1110110", buffer[i].find("11011")+1)!=-1 or buffer[i].find("0110111", buffer[i].find("11011")+1)!=-1 or buffer[i].find("1110111", buffer[i].find("11011")+1)!=-1)):
                        if buffer[i].find("11011", buffer[i].find("11011")+1) != -1:
                            restricted[1] = restricted[1]+1
                            #print(i, buffer[i].find("11011", buffer[i].find("11011")+1))
            if buffer[i].find("11101") != -1:
                restricted[1] = restricted[1] if(buffer[i].find("1111010")!=-1 or buffer[i].find("0111011")!=-1 or buffer[i].find("1111011")!=-1) else restricted[1]+1
        
        #print("四:{}".format(restricted[1]))
        if(restricted[1] > 1):
            return 3
        
        for i in range(4):
            if buffer[i].find("01110") != -1:
                restricted[0] = restricted[0] if(buffer[i].find("100111001")!=-1 or buffer[i].find("11101")!=-1 or buffer[i].find("10111")!=-1 or buffer[i].find("10011102")!=-1 or buffer[i].find("20111001")!=-1 or buffer[i].find("2011102")!=-1 or buffer[i].find("2011103")!=-1 or buffer[i].find("3011102")!=-1 or buffer[i].find("10011103")!=-1 or buffer[i].find("30111001")!=-1) else restricted[0]+1
            if buffer[i].find("010110") != -1:            
                restricted[0] = restricted[0] if(buffer[i].find("00101101")!=-1 or buffer[i].find("10101100")!=-1 or buffer[i].find("10101101")!=-1 or buffer[i].find("10101102")!=-1 or buffer[i].find("20101101")!=-1 or buffer[i].find("10101103")!=-1 or buffer[i].find("30101101")!=-1) else restricted[0]+1
            if buffer[i].find("011010") != -1:            
                restricted[0] = restricted[0] if(buffer[i].find("00110101")!=-1 or buffer[i].find("10110100")!=-1 or buffer[i].find("10110101")!=-1 or buffer[i].find("10110102")!=-1 or buffer[i].find("20110101")!=-1 or buffer[i].find("10110103")!=-1 or buffer[i].find("30110101")!=-1) else restricted[0]+1

        #print("三:{}".format(restricted[0]))
        #print()
        if(restricted[0] > 1):
            return 4

    def run(self):
        current_player = self.board.get_current_player()
        end, winner = self.board.game_end()
        
        if self.players[current_player].getStr() == "MCTS" and not end:
            self.cv.delete()
            player_in_turn = self.players[current_player]
            move = player_in_turn.get_action(self.board)
            i, j = self.board.move_to_location(move)
            if current_player==self.start_player.get_player_ind():
                availables = [i for i in self.board.availables]
                ban = self.forbidden(i,j,current_player)
                while ban == 1 or ban == 3 or ban == 4:
                    print("禁手!!")
                    print(i,j)
                    self.board.availables.remove(move)
                    move = player_in_turn.get_action(self.board)
                    i = move // self.board.width
                    j = move % self.board.width
                    ban = self.forbidden(i,j,current_player)
                self.board.availables = [i for i in availables]
                    
                if not self.top_of_cv == 0:
                    self.cv.delete(self.top_of_cv)
                self.cv.create_oval(self.chess_board_points[i][j].pixel_x-10, self.chess_board_points[i][j].pixel_y-10, self.chess_board_points[i][j].pixel_x+10, self.chess_board_points[i][j].pixel_y+10, fill='black') 
            else:
                if not self.top_of_cv == 0:
                    self.cv.delete(self.top_of_cv)
                self.cv.create_oval(self.chess_board_points[i][j].pixel_x-10, self.chess_board_points[i][j].pixel_y-10, self.chess_board_points[i][j].pixel_x+10, self.chess_board_points[i][j].pixel_y+10, fill='white')
            self.top_of_cv = self.cv.create_oval(self.chess_board_points[i][j].pixel_x-10, self.chess_board_points[i][j].pixel_y-10, self.chess_board_points[i][j].pixel_x+10, self.chess_board_points[i][j].pixel_y+10, outline='red', width=3)
            self.board.do_move(move)
        
        if end:
            if winner != -1:
                self.cv.create_text(self.board.width*15+15, self.board.height*30+30, text="Game over. Winner is {}".format(self.players[winner]))
                self.cv.unbind('<Button-1>')
            else:
                self.cv.create_text(self.board.width*15+15, self.board.height*30+30, text="Game end. Tie")

            return winner
        else:
            self.cv.after(100, self.run)
        
    def graphic(self, board):
        """
        Draw the board and show game info
        """
        width = board.width
        height = board.height

        window = tkinter.Tk()
        self.cv = tkinter.Canvas(window, height=height*30+60, width=width*30 + 30, bg = 'white')
        
        background = tkinter.PhotoImage(file="bg.gif")
        self.cv.create_image(0,0,anchor='nw',image=background)
        
        self.chess_board_points = [[None for i in range(height)] for j in range(width)]
        
        for i in range(width):
            for j in range(height):
                self.chess_board_points[i][j] = Point(i, j);
        for i in range(width):  #vertical line
            self.cv.create_line(self.chess_board_points[i][0].pixel_x, self.chess_board_points[i][0].pixel_y, self.chess_board_points[i][width-1].pixel_x, self.chess_board_points[i][width-1].pixel_y)
        
        for j in range(height):  #rizontal line
            self.cv.create_line(self.chess_board_points[0][j].pixel_x, self.chess_board_points[0][j].pixel_y, self.chess_board_points[height-1][j].pixel_x, self.chess_board_points[height-1][j].pixel_y)        
        
        self.button = tkinter.Button(window, text="start game!", command=self.run)
        self.cv.bind('<Button-1>', self.click1)
        self.cv.pack()
        self.button.pack()
        window.mainloop()
               
    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """
        start a game between two players
        """
        if start_player not in (0,1):
            raise Exception('start_player should be 0 (player1 first) or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        self.players = {p1: player1, p2:player2}
        self.start_player = self.players[start_player+1]
        
        if is_shown:
            self.graphic(self.board)
        else:
            while(1):
                current_player = self.board.get_current_player()
                player_in_turn = self.players[current_player]
                move = player_in_turn.get_action(self.board)
                i = move // self.board.width
                j = move % self.board.width                
                if current_player==self.start_player.get_player_ind():
                    availables = [i for i in self.board.availables]
                    ban = self.forbidden(i,j,current_player)
                    while ban == 1 or ban == 3 or ban == 4:
                        #print("禁手!!")
                        #print(i,j)
                        self.board.availables.remove(move)
                        move = player_in_turn.get_action(self.board)
                        i = move // self.board.width
                        j = move % self.board.width
                        ban = self.forbidden(i,j,current_player)
                    self.board.availables = [i for i in availables]

                self.board.do_move(move)
                if is_shown:
                    self.graphic(self.board)
                end, winner = self.board.game_end()
                if end:
                    return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree
        store the self-play data: (state, mcts_probs, z)
        """
        #print("start_self_play:")
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while(1):
            current_player = self.board.get_current_player()
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            i = move // self.board.width
            j = move % self.board.width
            #print(current_player)
            if current_player == 1:
                availables = [i for i in self.board.availables]
                ban = self.forbidden(i,j,current_player)
                while ban == 1 or ban == 3 or ban == 4:
                    #print("禁手!!")
                    #print(i,j)
                    self.board.availables.remove(move)
                    move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
                    i = move // self.board.width
                    j = move % self.board.width
                    ban = self.forbidden(i,j,current_player)
                self.board.availables = [i for i in availables]
            
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move   
            self.board.do_move(move)      
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))  
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                #reset MCTS root node
                player.reset_player() 
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
            
