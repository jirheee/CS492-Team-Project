# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

import json
import torch
import random

from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from nn_architecture import get_PVN_from_uuid


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


def run(data,force_cpu = False):
    f = open(data, encoding='utf-8')
    data = json.loads(f.read())

    try:
        width = data["board"]["board_width"]
        height = data["board"]["board_height"]
        n_in_row = data["board"]["n_in_row"]
        board = Board(width=width, height=height, n_in_row=n_in_row)
        game = Game(board)

        # ############### human VS AI ###################
        
        player1_uuid = data["player1"]
        player1 = get_PVN_from_uuid(player1_uuid,"best",force_cpu) 
        human =  Human()

        # set start_player=0 for human first

        # For random starting player, use 
        random.seed();start_player = random.randrange(2)

        #start_player = 1
        game.start_play(human, player1, start_player=start_player, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-g", "--game_config", help = "Game configuration .json file path")
    parser.add_argument("-c","--cpu", action="store_true",help="Force to run on CPU, without cuda", default=False)
    args = parser.parse_args()
    
    data = args.game_config

    run(data,args.cpu)
