from game import Board, Game
from nn_architecture import Conv
from rl_algorithm import DQNPlayer


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 8, 8
    model_file = './model/current_dqn_conv.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        nn_architecture = Conv(width, height)
        ai_player = DQNPlayer(nn_architecture)
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, ai_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
