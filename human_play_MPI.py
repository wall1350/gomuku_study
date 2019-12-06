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

from game_board import Board
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorlayer import PolicyValueNet
from mpi4py import MPI
from collections import Counter
from GUI_v1_4 import GUI
import os
import traceback

# how  to run :
# mpiexec -np 2 python -u human_play_mpi.py

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
        except:
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

# init model here
# if you want to load different models in each rank,
# you can assign it here ,like
# if rank == 0 : model_file = '...'
# if rank == 1 : model_file = '...'
board= Board(width=width,height=height,n_in_row=n_in_row)
model_file='model_15_15_5/best_policy.model'
best_policy = PolicyValueNet(board_width=width,board_height=height,block=19,init_model=model_file,cuda=True)
alpha_zero_player = MCTSPlayer(policy_value_function=best_policy.policy_value_fn_random,
                         action_fc=best_policy.action_fc_test,
                         evaluation_fc=best_policy.evaluation_fc2_test,
                         c_puct=5,
                         n_playout=400,
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
            bcast_move,move_probs,visits = player1.get_action(board=board,is_selfplay=False,print_probs_value=False)
        # bcast the move to other ranks
        bcast_move = comm.bcast(bcast_move, root=0)
        # print('!'*10,rank,bcast_move)

        # human do move
        board.do_move(bcast_move)

        if rank == 0:
            # print move index
            print(board.move_to_location(bcast_move))
            if is_shown:
                graphic(board=board)

    while True:

        # reset the search tree
        player2.reset_player()
        # AI's turn
        if rank == 0:
            # print prior probabilities
            gather_move, move_probs, visits = player2.get_action(board=board,is_selfplay=False,print_probs_value=True)
        else:
            gather_move, move_probs, visits = player2.get_action(board=board, is_selfplay=False, print_probs_value=False)

        gather_move_list = comm.gather(gather_move, root=0)

        if rank == 0:
            # gather ecah rank's move and get the most selected one
            print('list is', gather_move_list)
            bcast_move = Counter(gather_move_list).most_common()[0][0]

        # bcast the move to other ranks
        bcast_move = comm.bcast(bcast_move, root=0)
        # print('!' * 10, rank, bcast_move)

        # AI do move
        board.do_move(bcast_move)
        # print('rank:', rank, board.availables)

        if rank == 0:
            print(board.move_to_location(bcast_move))
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
            bcast_move, move_probs, visits = player1.get_action(board=board)

        # bcast the move to other ranks
        bcast_move = comm.bcast(bcast_move, root=0)
        # print('!'*10,rank,bcast_move)

        # human do move
        board.do_move(bcast_move)
        # print('rank:', rank, board.availables)

        if rank == 0:
            print(board.move_to_location(bcast_move))
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
    return 0

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

    gather_move_list,visits_list=[],[]
    tlist,tlist2=[],[]
    same_f_move = []
    ban,count = -1,0
    ins = ''

    while True:
        try:
            #if rank==0:
            #print('rank:',rank, ', loop start')
            if rank == 0:
                if current_player_num == 0:
                    UI.show_messages('Your turn')
                else:
                    UI.show_messages('AI\'s turn')
                    
                for move_availables in board.availables:
                    i, j = board.move_to_location(move_availables)
                    ban = forbidden(i, j)
        
                    if (ban == 1 or ban == 3 or ban == 4) and not end:
                        #print("ban at ",move_availables)
                        UI._draw_ban((i, j))
    
            # AI's turn
            if current_player_num == 1 and not end:
                print('***** rank:{}, enter while *****'.format(rank))
                availables = [i for i in board.availables]
                
                # reset the search tree
                player2.reset_player()
                while True:
                    #print(os.getpid(),"rank:",rank,", 1:",board.move_to_location(bcast_move), bcast_move, type(bcast_move))
                    if rank == 0:
                        # print prior probabilities
                        gather_move, move_probs, visits = player2.get_action(board=board, is_selfplay=False, print_probs_value=True)
                    else:
                        gather_move, move_probs, visits = player2.get_action(board=board, is_selfplay=False, print_probs_value=False)
                    
                    #print(os.getpid(),"rank: {0}, bef tlist:{1}, gather_move:{2}".format(rank,tlist,gather_move))
                    tlist = comm.gather(gather_move, root=0)
                    tlist2 = comm.gather(visits, root=0)
                    #print(os.getpid(),"rank: {0}, aft tlist:{1}, gather_move:{2}".format(rank,tlist,gather_move))
        
                    if rank == 0:
                        for i in tlist:
                            gather_move_list.append(i)
                        for i in tlist2:
                            visits_list.append(i)
                        print('rank: 0, gather_move_list:',gather_move_list)
                        print('rank: 0, visits_list:',visits_list)
                        c = Counter(gather_move_list).most_common()
                        if len(c)>1 and c[0][1]==c[1][1]:
                            same_f_move = [j[0] for j in c if j[1]==c[0][1]]
                            max_v_list=[0]*len(same_f_move)
                            for i in range(len(gather_move_list)):
                                for j in range(len(same_f_move)):
                                    if gather_move_list[i]==same_f_move[j] and max_v_list[j]<visits_list[i]:
                                        max_v_list[j] = visits_list[i]
                            bcast_move = same_f_move[max_v_list.index(max(max_v_list))]
                        else:
                            bcast_move = c[0][0]
                        print('rank: 0, bcast_move:',bcast_move)
                        gather_move_list.clear()
                        visits_list.clear()
                        tlist.clear()
                        tlist2.clear()
                        if current_player_num == SP:
                            x, y = board.move_to_location(bcast_move)
                            ban = forbidden(x, y)
                    
                    if rank==0:
                        if comm.Get_size()>1:
                            print('rank:{}, bef send ban:{}'.format(rank, ban))
                            for i in range(1,comm.Get_size()):
                                comm.send(ban,dest=i)
                            print('rank:{}, aft send ban:{}'.format(rank, ban))
                            #print('rank:{}, bef send bcast_move:{}'.format(rank, bcast_move))
                            for i in range(1,comm.Get_size()):
                                comm.send(bcast_move,dest=i)
                            #print('rank:{}, aft send bcast_move:{}'.format(rank, bcast_move))
                    else:
                        print('rank:{}, bef recv ban:{}'.format(rank, ban))
                        ban = comm.recv(source=0)
                        print('rank:{}, aft recv ban:{}'.format(rank, ban))
                        #print('rank:{}, bef recv bcast_move:{}'.format(rank, bcast_move))
                        bcast_move = comm.recv(source=0)
                        #print('rank:{}, aft recv bcast_move:{}'.format(rank, bcast_move))
                    
                    if ban == 0 or ban == 2:
                        break
                    elif ban == 1 or ban == 3 or ban == 4:
                        board.availables.remove(bcast_move)
                        
                board.availables = [i for i in availables]
                #print(os.getpid(),"rank: {0}, 2: {1}, {2}, {3}".format(rank,board.move_to_location(bcast_move),bcast_move,type(bcast_move)))
                print('***** rank:{}, end while *****'.format(rank))
            # human's turn
            else:
                if rank == 0:
                    inp = UI.get_input()
                    #print("inp={0}".format(inp))
                    if inp[0] == 'move' and not end:
                        count=0
                        if type(inp[1]) != int:
                            bcast_move = UI.loc_2_move(inp[1])
                        else:
                            bcast_move = inp[1]
                        if current_player_num == SP:
                            ban = forbidden(inp[2],inp[3])
                            if ban == 1 or ban == 3 or ban == 4:
                                continue
    
                    elif inp[0] == 'RestartGame':
                        UI.restart_game()
                        restart = 1
    
                    elif inp[0] == 'ResetScore':
                        UI.reset_score()
                        continue
    
                    elif inp[0] == 'quit':
                        restart = 'exit'
    
                    elif inp[0] == 'SwitchPlayer':
                        #print('******* SwitchPlayer *******')
                        SP = (SP + 1) % 2
                        UI.restart_game(False)
                        UI.reset_score()
                        restart = 1
                        ins = 'switch'
    
                    else:
                        print('ignored inp:', inp)
                        continue
                        
            #print('rank:{}, end:{}, restart:{}'.format(rank,end,restart))
            if not end and not restart:
                if rank == 0:
                    UI.render_step(bcast_move, board.current_player)
    
                #print(os.getpid(),"rank:",rank,", 3:",board.move_to_location(bcast_move), bcast_move,type(bcast_move))
                if rank==0:
                    board.do_move(bcast_move)
                    x, y = board.move_to_location(bcast_move)
                    print('last_move at: [{},{}]'.format(x,y))
                #print(os.getpid(),"rank:",rank,", 4:",board.move_to_location(bcast_move), bcast_move,type(bcast_move))
                
                current_player_num = (current_player_num + 1) % 2
                
                if rank==0:
                    if comm.Get_size()>1:
                        for i in range(1,comm.Get_size()):
                            #print('rank 0 is now send ins: move')
                            comm.send('move',dest=i)
                        for i in range(1,comm.Get_size()):
                            #print('rank 0 is now send board')
                            comm.send(board,dest=i)
                        #print('rank 0  aft send')
                    #print('rank {}:, board='.format(rank),board.states.keys())
                else:
                    #print('rank {} is now recv ins'.format(rank))
                    ins = comm.recv(source=0)
                    #print("rank {} aft recv ins:{}".format(rank, ins))
                    if ins=='move':
                        #print('rank {} is now recv move'.format(rank))
                        temp_board = comm.recv(source=0)
                        #print("rank {} aft recv temp_board:{}".format(rank, temp_board))
                        board.availables.clear()
                        board.states.clear()
                        board.states_sequence.clear()
                        for i in temp_board.availables:
                            board.availables.append(i)
                        for i in temp_board.states.keys():
                            board.states[i] = temp_board.states[i]
                        for i in temp_board.states_sequence:
                            board.states_sequence.append(i)
                        board.last_move = temp_board.last_move
                        board.current_player = temp_board.current_player
                        bcast_move = board.last_move
                        temp_board = None
                        #print('rank {}:, board='.format(rank),board.states.keys())
                    elif ins=='switch':
                        #print('rank {} is now recv board'.format(rank))
                        temp_board = comm.recv(source=0)
                        #print("rank {} aft recv temp_board:{}".format(rank, temp_board))
                        board.availables.clear()
                        board.states.clear()
                        board.states_sequence.clear()
                        for i in temp_board.availables:
                            board.availables.append(i)
                        for i in temp_board.states.keys():
                            board.states[i] = temp_board.states[i]
                        for i in temp_board.states_sequence:
                            board.states_sequence.append(i)
                        board.last_move = temp_board.last_move
                        board.current_player = temp_board.current_player
                        bcast_move = board.last_move
                        temp_board = None
                        current_player_num = SP = comm.recv(source=0)
                        player2.reset_player()

                end, winner = board.game_end()
                #print('rank:',rank,'end:', end)
    
                # check if game end
                if end:
                    if rank == 0:
                        if winner != -1:
                            print("Game end. Winner is ", winner)
                            UI.add_score(winner)
                        else:
                            print("Game end. Tie")
            else:
                #print('rank:{}, bef_bcast restart:{}'.format(rank,restart))
                temp_restart = comm.bcast(restart, root=0)
                #print('rank:{}, aft_bcast restart:{}'.format(rank,restart))
                if rank!=0 and type(temp_restart)==int:
                    restart = temp_restart
                elif rank!=0 and type(temp_restart)!=int and count==0:
                    count=1
                    continue
                #print('rank:{}, restart:{}'.format(rank,restart))
                if restart:
                    if restart == 'exit':
                        exit()
                    board.init_board()
                    current_player_num = SP
                    if rank==0 and ins=='switch':
                        #print("******* switch sending ********")
                        if comm.Get_size()>1:
                            for i in range(1,comm.Get_size()):
                                #print('rank 0 is now send switch')
                                comm.send(ins,dest=i)
                            #print('rank 0 aft send switch')
                            for i in range(1,comm.Get_size()):
                                #print('rank 0 is now send board')
                                comm.send(board,dest=i)
                            #print('rank 0 aft send board')
                            for i in range(1,comm.Get_size()):
                                #print('rank 0 is now send SP')
                                comm.send(SP,dest=i)
                            #print('rank 0 aft send SP')
                    player2.reset_player()
                    restart = 0
                    end = False
            #print('rank:',rank,'loop ends')
        except:
            print('rank:',rank,'error=',traceback.format_exc())

if __name__ == '__main__':
    start_play_with_UI()
