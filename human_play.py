
from game import Board, Game

from tf_policy_value_net import PolicyValueNet
from mcts_alphaZero import MCTSPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None
    
    def set_player_ind(self, p):
        self.player = p

    def get_player_ind(self):
        return self.player
    
    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]  # for python3
            move = board.location_to_move(location)
        except:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move
    
    def getStr(self):
        return "Human"

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n_row = 5
    width, height = 15, 15

    try:
        board = Board(width=width, height=height, n_in_row=n_row)
        game = Game(board)
        
        ################ human VS AI ###################        
      
        best_policy = PolicyValueNet("0710")
        best_policy2 = PolicyValueNet("0710")
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)  # set larger n_playout for better performance
        mcts_player2 = MCTSPlayer(best_policy2.policy_value_fn, c_puct=5, n_playout=400)
        #pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=20000)
        
        human = Human()
        human2 = Human()
        
        game.start_play(mcts_player, mcts_player2, start_player=0, is_shown=1)
        #game.start_play(human, pure_mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')

if __name__ == '__main__':
    run()
   

