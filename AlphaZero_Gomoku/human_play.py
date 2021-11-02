from game import Board, Game
from nn_architecture import Conv
from rl_algorithm import DQNPlayer
import json
import os
import random


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


def run(model_file):
    random.seed()
    with open('./data/example.json', encoding='utf-8') as f:
        data = json.loads(f.read())
    n = data["board"]["n_in_row"]
    width, height = data["board"]["board_width"], data["board"]["board_height"]
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        nn_architecture = lambda w,h: Conv(w, h)
        # AI player is better to be purely greedy when not training
        ai_player = DQNPlayer(nn_architecture, board_size = (width,height), eps=0)
        # Display which which model chekcpoint is being used.
        # Use default initialization when it could not be found
        if os.path.exists(model_file):
            print("using: ", model_file)
            ai_player.load_model(model_file)
        else:
            print(model_file, "does not exist")
            print("USING DEFAULT WEIGHT: Agent might act randomly")
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, ai_player, start_player=random.randrange(2), is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run(f'./model/best_dqn_conv_9by9_1.model')
