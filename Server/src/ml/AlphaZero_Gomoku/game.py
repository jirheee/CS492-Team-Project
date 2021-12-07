# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""
import random

import numpy as np

import json
from mcts_alphaZero import MCTSPlayer
from nn_architecture import get_PVN_from_uuid
import random

import os

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        cnn state shape: 4*width*height
        """
        cnn_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            cnn_state[0][move_curr // self.width,
                            move_curr % self.width] = 1.0
            cnn_state[1][move_oppo // self.width,
                            move_oppo % self.width] = 1.0
            # indicate the last move location
            cnn_state[2][self.last_move // self.width,
                            self.last_move % self.width] = 1.0
        if len(self.states) % 2 == 0:
            cnn_state[3][:, :] = 1.0  # indicate the colour to play

        return cnn_state

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n', flush = True)

    def start_play(self, player1, player2, start_player=0, is_shown=1, journal = None, output_path = None):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if journal:
            # initiallize record
            battle_record = {"starter":int(start_player+1),"moves":[],"winner":""}
        # First Move: Random
        w = self.board.width
        h = self.board.height
        if isinstance(players[self.board.get_current_player()],MCTSPlayer): # probably change this to detect if it is ai or not
            first_move = random.sample([w*(h//2)-w//2-1, w*(h//2)-w//2,w*(h//2)-w//2+1, w*(h//2)+w//2-1, w*(h//2)+w//2,w*(h//2)+w//2+1], 1)[0]
            self.board.do_move(first_move)
            if journal:
                battle_record["moves"].append((int(first_move//w),int(first_move%w)))
                print(f"moves: [{start_player+1}, ({first_move//w},{first_move%w})]",flush=True)
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if journal:
                battle_record["moves"].append((int(move//w), int(move%w)))
                print(f"moves: [{current_player}: ({move//w},{move%w})]",flush=True)
            if is_shown:
                # Display how random the policy is (eps): 0 is greedy, 1 is pure random
                print(move//self.board.width, move%self.board.width)
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if journal:
                    battle_record["winner"]=int(winner)
                    journal["battle_records"].append(battle_record)
                    print(f"winner: {current_player}",flush=True)
                    json.dump(journal,open(output_path,"w"))
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            current_state = self.board.current_state()
            states.append(current_state)
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



if __name__ == "__main__":
    import argparse
    import re,datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game_config", help = "Game configuration .json file path")
    parser.add_argument("-R", "--rounds", type = int, help="How many rounds do you want to play?", default = 1)
    parser.add_argument("-c","--cpu", action="store_true",help="Force to run on CPU, without cuda", default=False)
    args = parser.parse_args()

    f = open(args.game_config, encoding='utf-8')
    data = json.loads(f.read())
    
    width = data["board"]["board_width"]
    height = data["board"]["board_height"]
    n_in_row = data["board"]["n_in_row"]
    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)

    player1_uuid = data["player1"]
    player1 = get_PVN_from_uuid(player1_uuid,"best",args.cpu)

    player2_uuid = data["player2"]   
    player2 = get_PVN_from_uuid(player2_uuid,"best",args.cpu)
    random.seed()
    timestamp = re.sub(r'[^\w\-_\. ]', '_', datetime.datetime.now().__str__()[2:-7])
    json_output={"battle_records":[]} #[{"starter":0,"moves":[],"winner":1}]}
    player1_shorthand = player1_uuid.split("-")[0]
    player2_shorthand = player2_uuid.split("-")[0]
    output_path = f"../battle_records/{timestamp}-{player1_shorthand}-{player2_shorthand}.json"
    if not os.path.isdir("../battle_records"):
        os.mkdir("../battle_records")
    for ii in range(1,args.rounds+1):
        game.start_play(player1,player2,random.randrange(2), is_shown = 0, journal = json_output, output_path=output_path)
        #print(f"Round {ii} ended",end = "\n\n")