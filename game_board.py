# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:14:24 2018

@author: initial-h
"""

import numpy as np
from collections import deque
from GUI_v1_4 import GUI
import pygame
from pygame.locals import *

class Board(object):
    '''
    board for the game
    '''
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 11))
        self.height = int(kwargs.get('height', 11))
        self.states = {}
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        # need how many pieces in a row to win
        self.players = [1, 2]
        # player1 and player2

        self.feature_planes = 8
        # how many binary feature planes we use,
        # in alphago zero is 17 and the input to the neural network is 19x19x17
        # here is a (self.feature_planes+1) x self.width x self.height binary feature planes,
        # the self.feature_planes is the number of history features
        # the additional plane is the color feature that indicate the current player
        # for example, in 11x11 board, is 11x11x9,8 for history features and 1 for current player
        self.states_sequence = deque(maxlen=self.feature_planes)
        self.states_sequence.extendleft([[-1,-1]] * self.feature_planes)
        #use the deque to store last 8 moves
        # fill in with [-1,-1] when one game start to indicate no move

    def init_board(self, start_player=0):
        '''
        init the board and set some variables
        '''
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        self.availables = list(range(self.width * self.height))
        # keep available moves in a list
        # once a move has been played, remove it right away
        self.states = {}
        self.last_move = -1

        self.states_sequence = deque(maxlen=self.feature_planes)
        self.states_sequence.extendleft([[-1, -1]] * self.feature_planes)

    def move_to_location(self, move):
        '''
        transfer move number to coordinate

        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        '''
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        '''
        transfer coordinate to move number
        '''
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        '''
        return the board state from the perspective of the current player.
        state shape: (self.feature_planes+1) x width x height
        '''
        square_state = np.zeros((self.feature_planes+1, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            # states contain the (key,value) indicate (move,player)
            # for example
            # self.states.items() get dict_items([(1, 1), (2, 1), (3, 2)])
            # zip(*) get [(1, 2, 3), (1, 1, 2)]
            # then np.array and get
            # moves = np.array([1, 2, 3])
            # players = np.array([1, 1, 2])
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]

            # to construct the binary feature planes as alphazero did
            for i in range(self.feature_planes):
                # put all moves on planes
                if i%2 == 0:
                    square_state[i][move_oppo // self.width,move_oppo % self.height] = 1.0
                else:
                    square_state[i][move_curr // self.width,move_curr % self.height] = 1.0
            # delete some moves to construct the planes with history features
            for i in range(0,len(self.states_sequence)-2,2):
                for j in range(i+2,len(self.states_sequence),2):
                    if self.states_sequence[i][1]!= -1:
                        assert square_state[j][self.states_sequence[i][0] // self.width,self.states_sequence[i][0] % self.height] == 1.0, 'wrong oppo number'
                        square_state[j][self.states_sequence[i][0] // self.width, self.states_sequence[i][0] % self.height] = 0.
            for i in range(1,len(self.states_sequence)-2,2):
                for j in range(i+2,len(self.states_sequence),2):
                    if self.states_sequence[i][1] != -1:
                        assert square_state[j][self.states_sequence[i][0] // self.width,self.states_sequence[i][0] % self.height] ==1.0, 'wrong player number'
                        square_state[j][self.states_sequence[i][0] // self.width, self.states_sequence[i][0] % self.height] = 0.

        if len(self.states) % 2 == 0:
            # if %2==0，it's player1's turn to player,then we assign 1 to the the whole plane,otherwise all 0
            square_state[self.feature_planes][:, :] = 1.0  # indicate the colour to play

        # we should reverse it before return,for example the board is like
        # 0,1,2,
        # 3,4,5,
        # 6,7,8,
        # we will change it like
        # 6 7 8
        # 3 4 5
        # 0 1 2
        return square_state[:, ::-1, :]

    def do_move(self, move):
        '''
        update the board
        '''
        # print(self.states,move,self.current_player,self.players)
        self.states[move] = self.current_player
        # save the move in states
        self.states_sequence.appendleft([move,self.current_player])
        # save the last some moves in deque，so as to construct the binary feature planes
        self.availables.remove(move)
        #remove the played move from self.availables
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        # change the current player
        self.last_move = move

    def has_a_winner(self):
        '''
        judge if there's a 5-in-a-row, and which player if so
        '''
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
        '''
        Check whether the game is end
        '''
        end, winner = self.has_a_winner()
        if end:
            # if one win,return the winner
            return True, winner
        elif not len(self.availables):
            # if the board has been filled and no one win ,then return -1
            return True, -1
        return False, -1

    def get_current_player(self):
        '''
        return current player
        '''
        return self.current_player

class Game(object):
    '''
    game server
    '''
    def __init__(self, board, **kwargs):
        '''
        init a board
        '''
        self.board = board

    def forbidden(self, h, w, player=1):
        height = self.board.height
        width = self.board.width
        states = self.board.states
        n = self.board.n_in_row
        restricted = [0,0] # 33, 44
        buffer = ["33333333333" for j in range(4)] # 3: 棋盤外
        for i in range(w - n, w + n + 1): #下至上
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
        for count in range(11): #左上至右下
            i += 1
            j -= 1
            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            buffer[2] = buffer[2][:count] + str(states.get(i * width + j, 0)) + buffer[2][count+1:]
            buffer[2] = buffer[2][:5] + str(player) + buffer[2][6:]
        i = h - n - 1
        j = w - n - 1
        for count in range(11): #左下至右上
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
            before = restricted[1]
            restricted[1] = restricted[1]+1 if(buffer[i].find("011110")!=-1 or buffer[i].find("011112")!=-1 or buffer[i].find("211110")!=-1 or buffer[i].find("011113")!=-1 or buffer[i].find("311110")!=-1) else restricted[1]
            #if buffer[i].find("10111") != -1:
            #    restricted[1] = restricted[1] if(buffer[i].find("1101110")!=-1 or buffer[i].find("0101111")!=-1 or buffer[i].find("1101111")!=-1) else restricted[1]+1
            restricted[1] = restricted[1]+1 if(buffer[i].find("10111") != -1) else restricted[1]
            if buffer[i].find("11011") != -1:
                if(not (buffer[i].find("1110110")!=-1 or buffer[i].find("0110111")!=-1 or buffer[i].find("1110111")!=-1)):
                    restricted[1] = restricted[1]+1
                    #print(i, buffer[i].find("11011"))
                    if(not (buffer[i].find("1110110", buffer[i].find("11011")+1)!=-1 or buffer[i].find("0110111", buffer[i].find("11011")+1)!=-1 or buffer[i].find("1110111", buffer[i].find("11011")+1)!=-1)):
                        if buffer[i].find("11011", buffer[i].find("11011")+1) != -1:
                            restricted[1] = restricted[1]+1
                            #print(i, buffer[i].find("11011", buffer[i].find("11011")+1))
            #if buffer[i].find("11101") != -1:
            #    restricted[1] = restricted[1] if(buffer[i].find("1111010")!=-1 or buffer[i].find("0111011")!=-1 or buffer[i].find("1111011")!=-1) else restricted[1]+1
            restricted[1] = restricted[1]+1 if(buffer[i].find("11101") != -1) else restricted[1]
            if restricted[1] - before > 1:
                restricted[1] = restricted[1] if (buffer[i].find("111010111")!=-1 or buffer[i].find("11011011")!=-1 or buffer[i].find("1011101")!=-1) else restricted[1]-1
                if restricted[1] > 1:
                    return 3
        #print("四:{}".format(restricted[1]))
        if restricted[1] > 1:
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

    def graphic(self, board, player1, player2):
        '''
        Draw the board and show game info
        '''
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print(board.states)
        print()
        print(' ' * 2, end='')
        # rjust()
        # http://www.runoob.com/python/att-string-rjust.html
        for x in range(width):
            print("{0:4}".format(x), end='')
        # print('\r\n')
        print('\r')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(4), end='')
                elif p == player2:
                    print('O'.center(4), end='')
                else:
                    print('-'.center(4), end='')
            # print('\r\n') # new line
            print('\r')

    def start_play(self, player1, player2, start_player=0, is_shown=1,print_prob=True):
        '''
        start a game between two players
        '''
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        # print(p1,p2)
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move,move_probs = player_in_turn.get_action(self.board,is_selfplay=False,print_probs_value=print_prob)

            if current_player==start_player+1:
                availables = [i for i in self.board.availables]
                i, j = self.board.move_to_location(move)
                ban = self.forbidden(i, j)
                while ban == 1 or ban == 3 or ban == 4:
                    print("禁手!!", i, j)
                    self.board.availables.remove(move)
                    move, move_probs = AI.get_action(self.board, is_selfplay=False, print_probs_value=1)
                    i, j = self.board.move_to_location(move)
                    ban = self.forbidden(i, j)
                self.board.availables = [i for i in availables]

            self.board.do_move(move)

            if is_shown:
                print('player %r move : %r' % (current_player, [move // self.board.width, move % self.board.width]))
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()

            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_play_with_UI(self, AI, start_player=0):
        '''
        a GUI for playing
        '''
        AI.reset_player()
        self.board.init_board()
        current_player = SP = start_player+1
        UI = GUI(self.board.width)
        UI.SP = SP
        end = False
        while True:

            if SP == 0:
                UI._draw_text("Human(black)", (765, 150), text_height=UI.TestSize)
                UI._draw_text("AlphaZero(white)", (925, 150), text_height=UI.TestSize)
            else:
                UI._draw_text("AlphaZero(black)", (775, 150), text_height=UI.TestSize)
                UI._draw_text("Human(white)", (925, 150), text_height=UI.TestSize)

            print('current_player', current_player)
            if current_player == 1:
                UI.show_messages('Your turn')
            else:

                UI.show_messages('AI\'s turn')

            if SP == 1:
                for move_availables in self.board.availables:
                    i, j = self.board.move_to_location(move_availables)
                    ban = self.forbidden(i, j)

                    if ban == 1 or ban == 3 or ban == 4:
                        print("ban at ",move_availables)
                        UI._draw_ban((i, j))

            if current_player == 2 and not end:
                move, move_probs = AI.get_action(self.board, is_selfplay=False, print_probs_value=1)
                if current_player == SP:
                    availables = [i for i in self.board.availables]


                    i, j = self.board.move_to_location(move)
                    ban = self.forbidden(i, j)
                    while ban == 1 or ban == 3 or ban == 4:
                        print("禁手!!", i, j)
                        self.board.availables.remove(move)
                        move, move_probs = AI.get_action(self.board, is_selfplay=False, print_probs_value=1)
                        i, j = self.board.move_to_location(move)
                        ban = self.forbidden(i, j)
                    self.board.availables = [i for i in availables]
            else:
                inp = UI.get_input()
                if inp[0] == 'move' and not end:
                    move = inp[1]
                    if current_player == SP:
                        ban = self.forbidden(inp[2],inp[3])
                        if ban == 1 or ban == 3 or ban == 4:
                            continue
                elif inp[0] == 'RestartGame':
                    end = False
                    current_player = SP
                    self.board.init_board()
                    UI.restart_game()
                    AI.reset_player()
                    continue
                elif inp[0] == 'ResetScore':
                    UI.reset_score()
                    continue
                elif inp[0] == 'quit':
                    exit()
                    continue
                elif inp[0] == 'SwitchPlayer':
                    end = False
                    self.board.init_board()
                    UI.restart_game(False)
                    UI.reset_score()
                    AI.reset_player()
                    SP = self.board.players[0] if SP == self.board.players[1] else self.board.players[1]
                    current_player = SP
                    UI.SP=SP

                    continue
                else:
                    # print('ignored inp:', inp)
                    continue
            # print('player %r move : %r'%(current_player,[move//self.board.width,move%self.board.width]))
            if not end:
                # print(move, type(move), current_player)
                UI.render_step(move, self.board.current_player)
                self.board.do_move(move)
                UI.show_messages('AI\'s turn')
                # print('move', move)
                # print(2, self.board.get_current_player())
                current_player = self.board.players[0] if current_player == self.board.players[1] else self.board.players[1]
                # UI.render_step(move, current_player)
                end, winner = self.board.game_end()
                if end:
                    #UI.show_dialog() 預備給對話視窗用
                    if winner != -1:
                        print("Game end. Winner is player", winner)
                        UI.add_score(winner)
                        UI._clean_ban()

                        if winner == 1:
                            #UI.show_messages("Game end. Winner is 1 ")
                            UI._draw_text("Game end. Winner is  player 1 ", (500, 690), text_height=UI.TestSize)
                        elif winner == 2:
                            #UI.show_messages("Game end. Winner is 2 ")
                            UI._draw_text("Game end. Winner is player 2 ", (500, 690), text_height=UI.TestSize)
                    else:
                        print("Game end. Tie")
                        UI._draw_text("Game end. Tie ", (775, 750), text_height=UI.TestSize)
                    #print(UI.score)
                    #print()

    def start_self_play(self, player, is_shown=0):
        '''
        start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        '''
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 is_selfplay=True,
                                                 print_probs_value=False)

            if current_player == 1:
                availables = [i for i in self.board.availables]
                i, j = self.board.move_to_location(move)
                ban = self.forbidden(i, j)
                while ban == 1 or ban == 3 or ban == 4:
                    self.board.availables.remove(move)
                    move, move_probs = player.get_action(self.board,
                                                         is_selfplay=True,
                                                         print_probs_value=False)
                    i, j = self.board.move_to_location(move)
                    ban = self.forbidden(i, j)
                self.board.availables = [i for i in availables]

            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
