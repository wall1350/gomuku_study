# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:32:27 2018

@author: Tokarev-TT-33

introduction:
    This is a UI for Gomoku game which does not contain the rule code.

    Set self.UnitSize in __init__() to a different value can change the basic size of all elements

    There are no limits for the value of board_size. So a board of any size can be created(if the system supports).
    Add limits in self.reset() if necessary.
"""
import pygame
from pygame.locals import *



class Chess(pygame.sprite.Sprite):
    def __init__(self,player,pos):
        pygame.sprite.Sprite.__init__(self)
        self.player=player
        self.pos=pos


class GUI:

    def __init__(self, board_size=15):
        pygame.init()
        self.SP = 0
        self.all_chess_list = pygame.sprite.Group()
        self.score = [0, 0]
        self.BoardSize = board_size
        self.UnitSize = 40      # the basic size of all elements, try a different value!
        self.TestSize = int(self.UnitSize * 0.5)
        self.state = {}         # a dictionary for pieces on board. filled with move-player pairs, such as 34:1
        self.areas = {}         # a dictionary for button areas. filled with name-Rect pairs
        self.ScreenSize = None  # save the screen size for some calculation
        self.screen = None
        self.last_action_player = None
        self.round_counter = 0
        self.messages = ''
        """self._background_color = (197, 227, 205)"""
        self._background_color = (247, 226, 193)
        self._board_color = (254, 185, 120)
        self._button_color = (245, 154 , 35)

        self.reset(board_size)

        # restart_game() must be called before reset_score() because restart_game() will add value to self.round_counter
        self.restart_game(False)
        self.reset_score()
        #self._draw_text("Human(black)", (765, 150), text_height=self.TestSize)
        #self._draw_text("AlphaZero(white)", (925, 150), text_height=self.TestSize)

    def reset(self, bs):
        """
        reset screen
        :param bs: board size
        """

        # # you can add limits for board size
        # bs = int(bs)
        # if bs < 5:
        #     raise ValueError('board size too small')

        self.BoardSize = bs
        self.ScreenSize = (1020,710)
        print("ScreenSize",self.ScreenSize)
        self.screen = pygame.display.set_mode(self.ScreenSize, 0, 32)
        pygame.display.set_caption('AlphaZero_Gomoku')

        # button areas
        self.areas['SwitchPlayer'] = Rect(704, 640, self.UnitSize*2, self.UnitSize)
        self.areas['ResetScore'] = Rect(810, 640, self.UnitSize*2, self.UnitSize)
        self.areas['RestartGame'] = Rect(916, 640, self.UnitSize*2, self.UnitSize)

        board_lenth = self.UnitSize * self.BoardSize
        self.areas['board'] = Rect(self.UnitSize, self.UnitSize, board_lenth, board_lenth)

    def restart_game(self, button_down=True):
        """
        restart for a new round
        :param button_down: whether the RestartGame button is pressed, used to highlight button.
        """
        self.round_counter += 1
        self.all_chess_list.empty()
        self._draw_static()
        if button_down:
            self._draw_button('RestartGame', 1)
        self.state = {}
        self.last_action_player = None
        pygame.display.update()

    def reset_score(self):
        """
        reset score and round
        """
        self.score = [0, 0]
        self.round_counter = 1
        self.show_messages()

    def add_score(self, winner):
        """
        add score for winner
        :param winner: the name of the winner
        """
        if winner == 1:
            self.score[0] += 1
        elif winner == 2:
            self.score[1] += 1
        else:
            raise ValueError('player number error')
        self.show_messages()

    def render_step(self, action, player):
        """
        render a step of the game
        :param action: 1*2 dimension location value such as (2, 3) or an int type move value such as 34
        :param player: the name of the player
        """
        try:
            action = int(action)
        except Exception:
            pass
        if type(action) != int:
            move = self.loc_2_move(action)
        else:
            move = action

        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

        if self.last_action_player:     # draw a cross on the last piece to mark the last move
            self._draw_pieces(self.last_action_player[0], self.last_action_player[1], False)

        self._draw_pieces(action, player, True)
        self.state[move] = player
        self.last_action_player = move, player

    def move_2_loc(self, move):
        """
        transfer a move value to a location value
        :param move: an int type move value such as 34
        :return: an 1*2 dimension location value such as (2, 3)
        """
        return move // self.BoardSize, move % self.BoardSize

    def loc_2_move(self, loc):
        """
        transfer a move value to a location value
        :param loc: an 1*2 dimension location value such as (2, 3)
        :return: an int type move value such as 34
        """
        return loc[0] * self.BoardSize + loc[1]

    def get_input(self):
        """
        get inputs from clicks
        :return: variable-length array.[0] is the name. Additional information behind (maybe not exist).
        """
        while True:
            event = pygame.event.wait()
            if event.type == QUIT:
                return 'quit',

            if event.type == MOUSEBUTTONDOWN:   # check mouse click event
                if event.button == 1:
                    mouse_pos = event.pos

                    for name, rec in self.areas.items():
                        if self._in_area(mouse_pos, rec):
                            if name != 'board':
                                self._draw_button(name, 2, True)
                                pygame.time.delay(100)
                                self._draw_button(name, 1, True)
                                return name,
                            else:
                                x = (mouse_pos[0] - self.UnitSize)//self.UnitSize
                                y = self.BoardSize - (mouse_pos[1] - self.UnitSize)//self.UnitSize - 1
                                move = self.loc_2_move((x, y))
                                if move not in self.state:
                                    return 'move', move, x, y
                                    #return 'move', move

            if event.type == MOUSEMOTION:       # check mouse move event to highlight buttons
                mouse_pos = event.pos
                for name, rec in self.areas.items():
                    if name != 'board':
                        if self._in_area(mouse_pos, rec):
                            self._draw_button(name, 1, True)
                        else:
                            self._draw_button(name, update=True)

    def deal_with_input(self, inp, player):
        """
        This is just a example to deal with inputs
        :param inp: inputs from get_input()
        :param player: the name of the player
        """
        if inp[0] == 'RestartGame':
            self.restart_game()
        elif inp[0] == 'ResetScore':
            self.reset_score()
        elif inp[0] == 'quit':
            exit()
        elif inp[0] == 'move':
            self.render_step(inp[1], player)
        elif inp[0] == 'SwitchPlayer':
            # restart_game() must be called before reset_score(). The reason is mentioned above.
            UI.restart_game(False)
            UI.reset_score()
            # code for switch is needed

    def show_messages(self, messages=None):
        """
        show extra messages on screen
        :param messages:
        :return:
        """
        if messages:
            self.messages = messages
        pygame.draw.rect(self.screen, self._background_color, (700, 80, 150, 20))
        self._draw_round(False)
        self._draw_text(self.messages, (745, 90), text_height=self.TestSize)
        self._draw_score()

    def _draw_score(self, update=True):
        score = 'Score: ' + str(self.score[0]) + ' : ' + str(self.score[1])
        self._draw_text(score, (940, 90),
                        backgroud_color=self._background_color, text_height=self.TestSize)
        if update:
            pygame.display.update()

    def _draw_round(self, update=True):
        self._draw_text('Round: ' + str(self.round_counter), (950, 205),
                        backgroud_color=self._background_color, text_height=self.TestSize)
        if update:
            pygame.display.update()

    def _draw_pieces(self, loc, player, last_step=False):

        """
        draw pieces
        :param loc:  1*2 dimension location value such as (2, 3) or an int type move value such as 34
        :param player: the name of the player
        :param last_step: whether it is the last step
        """
        try:
            loc = int(loc)
        except Exception:
            pass

        if type(loc) is int:
            x, y = self.move_2_loc(loc)
        else:
            x, y = loc
        pos = int(self.UnitSize * 1.5 + x * self.UnitSize), int(self.UnitSize * 1.5 + (self.BoardSize - y - 1) * self.UnitSize)
        if player == 1:
            c = (0, 0, 0)
        elif player == 2:
            self._clean_ban()
            c = (255, 255, 255)
        else:
            raise ValueError('num input ValueError')
        pygame.draw.circle(self.screen, c, pos, int(self.UnitSize * 0.45))

        chess = Chess(player, pos)
        self.all_chess_list.add(chess)
        #print("chess ",chess.player,chess.pos)

        if last_step:
            if player == 1:
                c = (255, 255, 255)
            elif player == 2:
                c = (0, 0, 0)

            start_p1 = pos[0] - self.UnitSize * 0.3, pos[1]
            end_p1 = pos[0] + self.UnitSize * 0.3, pos[1]
            pygame.draw.line(self.screen, c, start_p1, end_p1)

            start_p2 = pos[0], pos[1] - self.UnitSize * 0.3
            end_p2 = pos[0], pos[1] + self.UnitSize * 0.3
            pygame.draw.line(self.screen, c, start_p2, end_p2)

    def _clean_ban(self):
        print("_clean_ban")
        self._draw_static()
        for chess in self.all_chess_list.sprites():
            if chess.player == 1:
                c = (0, 0, 0)
            elif chess.player == 2:
                c = (255, 255, 255)
            else:
                raise ValueError('num input ValueError')
            pygame.draw.circle(self.screen, c, chess.pos, int(self.UnitSize * 0.45))


    def _draw_ban(self,loc):
        print("_draw_ban")

        ban_image = pygame.image.load("ban.png")
        ban_image = pygame.transform.scale(ban_image, (50, 50))
        ban_image.convert()
        x, y = loc
        pos = int(self.UnitSize * 0.9 + x * self.UnitSize), int(self.UnitSize * 0.9 + (self.BoardSize - y - 1) * self.UnitSize)
        #if player == 1:
        #ban = Ban("ban.png", pos)
        #self.all_ban_list.add(ban)
        #print("all_ban_list ",self.all_ban_list)
        #for ban in self.all_ban_list:
        self.screen.blit(ban_image,pos)
        pygame.display.update()

    def _show_endmsg(self,winner):
        print("_show_endmsg")
        p1_image = pygame.image.load("p1.png")
        p2_image = pygame.image.load("p2.png")
        tie_image = pygame.image.load("tie.png")
        if winner == 1:
            self.screen.blit(p1_image,(40, 280))
        elif winner == 2:
            self.screen.blit(p2_image,(40, 280))
        else:
            self.screen.blit(tie_image,(40, 280))
        pygame.display.update()

    def _draw_static(self):


        """
        Draw static elements that will not change in a round.
        """
        self.screen.fill(self._background_color)
        # load background picture
        background_image = pygame.image.load("bg.png")
        panel_image = pygame.image.load("panel.png")

        # scale up the size of background picture
        background_image = pygame.transform.scale(background_image, (680, 680))
        panel_image = pygame.transform.scale(panel_image, (300, 520))

        background_image.convert()
        panel_image.convert()

        # the first para is the picture item , the second one is the position in the flame
        self.screen.blit(background_image, (0, 0))
        self.screen.blit(panel_image, (700, 100))

        # draw board
        board_lenth = self.UnitSize * self.BoardSize
        #pygame.draw.rect(self.screen, self._board_color, self.areas['board'])
        for i in range(self.BoardSize):
            # draw grid lines
            start = self.UnitSize * (i + 0.5)
            pygame.draw.line(self.screen, (0, 0, 0), (start + self.UnitSize, self.UnitSize*1.5),
                             (start + self.UnitSize, board_lenth + self.UnitSize*0.5))
            pygame.draw.line(self.screen, (0, 0, 0), (self.UnitSize*1.5, start + self.UnitSize),
                             (board_lenth + self.UnitSize*0.5, start + self.UnitSize))
            pygame.draw.rect(self.screen, (0, 0, 0), (self.UnitSize, self.UnitSize, board_lenth, board_lenth), 1)
            # coordinate values
            self._draw_text(self.BoardSize - i - 1, (self.UnitSize / 2, start + self.UnitSize), text_height=self.TestSize)  # 竖的
            self._draw_text(i, (start + self.UnitSize, self.UnitSize / 2), text_height=self.TestSize)  # 横的

        # draw buttons
        for name in self.areas.keys():
            if name != 'board':
                self._draw_button(name)

        self.show_messages()

    def _draw_text(self, text, position, text_height=25, font_color=(0, 0, 0), backgroud_color=None, pos='center',angle=0):
        """
        draw text
        :param text: a string type text
        :param position: the location point
        :param text_height: text height
        :param font_color: font color
        :param backgroud_color: background color
        :param pos: the location point is where in the text rectangle.
        'center','top','bottom','left','right'and their combination such as 'topleft' can be selected
        :param angle: the rotation angle of the text
        """
        posx, posy = position
        font_obj = pygame.font.Font(None, int(text_height))
        text_surface_obj = font_obj.render(str(text), True, font_color, backgroud_color)
        text_surface_obj = pygame.transform.rotate(text_surface_obj, angle)
        text_rect_obj = text_surface_obj.get_rect()
        exec('text_rect_obj.' + pos + ' = (posx, posy)')
        self.screen.blit(text_surface_obj, text_rect_obj)

    def _draw_button(self, name, high_light=0, update=False):
        rec = self.areas[name]
        if not high_light:
            color = self._button_color
        elif high_light == 1:
            color = self._button_color
        elif high_light == 2:
            color = self._button_color
        else:
            raise ValueError('high_light value error')
        pygame.draw.rect(self.screen, color, rec)
        pygame.draw.rect(self.screen, (0, 0, 0), rec, 1)
        self._draw_text(name, rec.center, text_height=self.TestSize)
        if update:
            pygame.display.update()

    @staticmethod
    def _in_area(loc, area):
        """
        check whether the location is in area
        :param loc: a 1*2 dimension location value such as (123, 45)
        :param area: a Rect type value in pygame
        """
        return True if area[0] < loc[0] < area[0] + area[2] and area[1] < loc[1] < area[1] + area[3] else False


if __name__ == '__main__':
    # test
    UI = GUI()
    action = 22
    player = 1
    i = 1
    while True:
        """if i == 1:
            UI.show_messages('first player\'s turn')
        else:
            UI.show_messages('second player\'s turn')"""
        inp = UI.get_input()
        print(inp)
        UI.deal_with_input(inp, i)
        if inp[0] == 'move':
            i %= 2
            i += 1
        elif inp[0] == 'RestartGame':
            i = 1
        elif inp[0] == 'SwitchPlayer':
            i = 1
