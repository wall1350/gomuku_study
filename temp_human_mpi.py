# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:45:36 2018

@author: initial-h
"""
'''
write a root parallel mcts and vote a move like ensemble way
why i do this:
when i train the network, i should try some parameter settings
and train for a while to compare which is better,
so there are many semi-finished model get useless,it's waste of computation resource
even though i can continue to train based on the semi-finished model.
i write the parallel way using MPI,
so that each rank can load different model and then vote the next move to play, besides,
you can also weights each model to get the weighted next move(i don't do it here but it's easy to realize).

and also each rank can load the same model and vote the next move,
besides the upper benifit ,it can also improve the strength and save the playout time by parallel.
some other parallel ways can find in《Parallel Monte-Carlo Tree Search》.
'''
import pygame
from pygame.locals import *
from game_board import Board
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorlayer import PolicyValueNet
from mpi4py import MPI
from collections import Counter
from GUI_v1_4 import GUI
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# how  to run :
# mpiexec -np 2 python -u human_play_mpi.py

#win-loss record
win_loss_list=[]
show_start = 0
#　MPI setting
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# game setting
n_in_row = 5
width, height = 15, 15

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board,is_selfplay=False,print_probs_value=0):
        # no use params in the func : is_selfplay,print_probs_value
        # just to stay the same with AI's API
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move,_ = self.get_action(board)
        return move,None

    def __str__(self):
        return "Human {}".format(self.player)



def graphic(board, player1=1, player2=2):
    '''
    Draw the board and show game info
    '''
    width = board.width
    height = board.height

    print("Player", player1, "with X".rjust(3))
    print("Player", player2, "with O".rjust(3))
    print("board.states",board.states)
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


board= Board(width=width,height=height,n_in_row=n_in_row)

# init model here
# if you want to load different models in each rank,
# you can assign it here ,like
# if rank == 0 : model_file = '...'
# if rank == 1 : model_file = '...'

model_file='model_15_15_5/best_policy.model'
best_policy = PolicyValueNet(board_width=width,board_height=height,block=19,init_model=model_file,cuda=True)
alpha_zero_player = MCTSPlayer(policy_value_function=best_policy.policy_value_fn_random,
                         action_fc=best_policy.action_fc_test,
                         evaluation_fc=best_policy.evaluation_fc2_test,
                         c_puct=5,
                         n_playout=1,
                         is_selfplay=False)

player1 = Human()
player2 = alpha_zero_player
# player2 = MCTS_Pure(5,200)

def start_play(start_player=0, is_shown=1):
    # run a gomoku game with AI in terminal
    bcast_move = -1

    # init game and player
    board.init_board()
    player2.reset_player()

    end = False
    if rank == 0 and is_shown:
        # draw board in terminal
        graphic(board=board)

    if start_player == 0:
        # human first to play
        if rank == 0:
            bcast_move,move_probs = player1.get_action(board=board,is_selfplay=False,print_probs_value=False)
        # bcast the move to other ranks
        bcast_move = comm.bcast(bcast_move, root=0)
        # print('!'*10,rank,bcast_move)

        # human do move
        board.do_move(bcast_move)

        if rank == 0:
            # print move index
#            print(board.move_to_location(bcast_move))
            if is_shown:
                graphic(board=board)

    while True:

        # reset the search tree
        player2.reset_player()
        # AI's turn
        if rank == 0:
            # print prior probabilities
            gather_move, move_probs = player2.get_action(board=board,is_selfplay=False,print_probs_value=True)
        else:
            gather_move, move_probs = player2.get_action(board=board, is_selfplay=False, print_probs_value=False)

        gather_move_list = comm.gather(gather_move, root=0)

        if rank == 0:
            # gather ecah rank's move and get the most selected one
#            print('list is', gather_move_list)
            bcast_move = Counter(gather_move_list).most_common()[0][0]

        # bcast the move to other ranks
        bcast_move = comm.bcast(bcast_move, root=0)
        # print('!' * 10, rank, bcast_move)

        # AI do move
        board.do_move(bcast_move)
        # print('rank:', rank, board.availables)

        if rank == 0:
#            print(board.move_to_location(bcast_move))
            if is_shown:
                graphic(board=board)
        end, winner = board.game_end()

        # check if game end
        if end:
            if rank == 0:
                if winner != -1:
                    print("Game end. Winner is ", winner)
                else:
                    print("Game end. Tie")
            break

        # human's turn
        if rank == 0:
            bcast_move, move_probs = player1.get_action(board=board)

        # bcast the move to other ranks
        bcast_move = comm.bcast(bcast_move, root=0)
        # print('!'*10,rank,bcast_move)

        # human do move
        board.do_move(bcast_move)
        # print('rank:', rank, board.availables)

        if rank == 0:
#            print(board.move_to_location(bcast_move))
            if is_shown:
                graphic(board=board)
        end, winner = board.game_end()

        # check if game end
        if end:
            if rank == 0:
                if winner != -1:
                    print("Game end. Winner is ", winner)
                else:
                    print("Game end. Tie")
            break

def forbidden(h, w, player=1):
    height = board.height
    width = board.width
    states = board.states
    n = board.n_in_row
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


    for i in range(4):
        judge = 0
        for j in range(11):
            judge = judge+1 if(buffer[i][j]==str(player)) else 0
            if judge == 5:
                return 1 if (j<10 and buffer[i][j+1]==str(player)) else 2 # 1長連禁手  2 win

    for i in range(4):
        before = restricted[1]
        restricted[1] = restricted[1]+1 if(buffer[i].find("011110")!=-1 or buffer[i].find("011112")!=-1 or buffer[i].find("211110")!=-1 or buffer[i].find("011113")!=-1 or buffer[i].find("311110")!=-1) else restricted[1]

        restricted[1] = restricted[1]+1 if(buffer[i].find("10111") != -1) else restricted[1]
        if buffer[i].find("11011") != -1:
            if(not (buffer[i].find("1110110")!=-1 or buffer[i].find("0110111")!=-1 or buffer[i].find("1110111")!=-1)):
                restricted[1] = restricted[1]+1
                #print(i, buffer[i].find("11011"))
                if(not (buffer[i].find("1110110", buffer[i].find("11011")+1)!=-1 or buffer[i].find("0110111", buffer[i].find("11011")+1)!=-1 or buffer[i].find("1110111", buffer[i].find("11011")+1)!=-1)):
                    if buffer[i].find("11011", buffer[i].find("11011")+1) != -1:
                        restricted[1] = restricted[1]+1

        restricted[1] = restricted[1]+1 if(buffer[i].find("11101") != -1) else restricted[1]
        """print('before: ', restricted[1])"""
        if restricted[1] - before > 1:
            restricted[1] = restricted[1] if (buffer[i].find("111010111")!=-1 or buffer[i].find("11011011")!=-1 or buffer[i].find("1011101")!=-1) else 1
            if restricted[1] > 1:
                return 3
        """print('after: ', restricted[1])"""

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

def start_play_with_UI(start_player=0):
    # run a gomoku game with AI in GUI
    bcast_move = -1

    # init game and player
    board.init_board()
    player2.reset_player()

    current_player_num = start_player
    restart = 0
    end = False
    if rank == 0:
        SP = start_player
        UI = GUI(board.width)

    while True:

        if rank == 0:

            for i in range(show_start, show_start+8, 1):

                if (i <len(win_loss_list)):
                    UI._draw_text(win_loss_list[i][0], (730, 305+i*40),backgroud_color=(255,255,255), text_height=30)
                    UI._draw_text(win_loss_list[i][1], (820, 305+i*40),backgroud_color=(255,255,255), text_height=30)
                    UI._draw_text(win_loss_list[i][2], (930, 305+i*40),backgroud_color=(255,255,255), text_height=30)
                    pygame.draw.line(UI.screen, UI._button_color, (705, 320+i*40), (995, 320+i*40),1)


            for move_availables in board.availables:
                i, j = board.move_to_location(move_availables)
                ban = forbidden(i, j)

                if (ban == 1 or ban == 3 or ban == 4) and not end:
                    print("ban at ",move_availables)
                    UI._draw_ban((i, j))

            if current_player_num == 0:
                UI.show_messages('Your turn')
            else:
                UI.show_messages('AI\'s turn')

        # AI's turn
        if current_player_num == 1 and not end:
            # reset the search tree
            player2.reset_player()
            if rank == 0:
                # print prior probabilities
                gather_move, move_probs = player2.get_action(board=board, is_selfplay=False, print_probs_value=True)

            else:
                gather_move, move_probs = player2.get_action(board=board, is_selfplay=False, print_probs_value=False)

            gather_move_list = comm.gather(gather_move, root=0)
            # print('list is', gather_move_list)

            if rank == 0:
                # gather ecah rank's move and get the most selected one
#                print('list is', gather_move_list)
                bcast_move = Counter(gather_move_list).most_common()[0][0]
                # print(board.move_to_location(bcast_move))

        # human's turn
        else:
            if rank == 0:
                inp = UI.get_input()
                if inp[0] == 'move' and not end:
                    if type(inp[1]) != int:
                        bcast_move = UI.loc_2_move(inp[1])
                    else:
                        bcast_move = inp[1]

                elif inp[0] == 'RestartGame':
                    UI.restart_game()
                    restart = SP+1

                elif inp[0] == 'ResetScore':
                    UI.reset_score()
                    continue

                elif inp[0] == 'quit':
                    restart = 'exit'

                elif inp[0] == 'SwitchPlayer':
                    SP = (SP + 1) % 2
                    UI.restart_game(False)
                    UI.reset_score()
                    win_loss_list=[]
                    restart = SP+1
                elif inp[0] == 'NextPage':
                    if show_start + 8 < len(win_loss_list):
                        show_start = show_start+8
                        continue
                elif inp[0] == 'LastPage':
                    if show_start - 8 > 0:
                        show_start = show_start-8
                        continue
                else:
                    # print('ignored inp:', inp)
                    continue

        restart = comm.bcast(restart, root=0)

        if not end and not restart:
            # bcast the move to other ranks
            bcast_move = comm.bcast(bcast_move, root=0)
            # print('!'*10,rank,bcast_move)
            if rank == 0:
#                print("board.move_to_location",board.move_to_location(bcast_move))
                UI.render_step(bcast_move, board.current_player)

            print('bcast_move',board.current_player, bcast_move)
            # human do move
            board.do_move(bcast_move)


            current_player_num = (current_player_num + 1) % 2
            end, winner = board.game_end()

            # check if game end
            if end:
                if rank == 0:
                    if winner != -1:
                        #print("Game end. Winner is ", winner)
                        UI.add_score(winner)
                        UI._show_endmsg(winner) #結束提示

                        now_list=[]
                        now_list.append(UI.round_counter)
                        now_list.append(str(winner))
                        now_list.append('12:87:23')
                        win_loss_list.append(now_list)
                        print(win_loss_list)

                        for i in range(show_start, show_start+8, 1):

                            if (i <len(win_loss_list)):
                                UI._draw_text(win_loss_list[i][0], (730, 305+i*40),backgroud_color=(255,255,255), text_height=30)
                                UI._draw_text(win_loss_list[i][1], (820, 305+i*40),backgroud_color=(255,255,255), text_height=30)
                                UI._draw_text(win_loss_list[i][2], (930, 305+i*40),backgroud_color=(255,255,255), text_height=30)
                                pygame.draw.line(UI.screen, UI._button_color, (705, 320+i*40), (995, 320+i*40),1)

                    else:
                        print("Game end. Tie")

        else:
            if restart:
                if restart == 'exit':
                    exit()
                board.init_board()
                player2.reset_player()
                current_player_num = restart-1
                restart = 0
                end = False


if __name__ == '__main__':
    # start_play(start_player=0,is_shown=True)

    start_play_with_UI()
