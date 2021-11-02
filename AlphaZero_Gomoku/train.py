import json
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
from game import Board, Game
from nn_architecture import Conv
from rl_algorithm import DQNPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure

import re
import datetime


class TrainPipeline():
    def __init__(self, data='./data/example.json'):
        # load data from json file
        f = open(data, encoding='utf-8')
        data = json.loads(f.read())

        # params of the board and the game
        self.board_width = data["board"]["board_width"]
        self.board_height = data["board"]["board_height"]
        self.n_in_row = data["board"]["n_in_row"]
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # training params
        self.lr = data["hyperparameters"]["lr"]
        self.buffer_size = data["hyperparameters"]["buffer_size"]
        self.batch_size = data["hyperparameters"]["batch_size"]
        self.epochs = data["hyperparameters"]["epochs"]
        self.data_buffer = deque(maxlen=self.buffer_size)

        # Neural Network Architecture
        if data["nn_architecture"] == 'Conv':
            # Instead of NN model instance, use constructor
            self.nn_architecture = lambda w,h:Conv(w,h)

        # Algorithm
        if data["rl_algorithm"] == 'DQN':
            # To use the constructor inside, board size should be passed on
            self.player = DQNPlayer(self.nn_architecture, lr=self.lr, board_size = (self.board_width, self.board_height))

        self.check_freq = 5000 # Checkpoint for evaluation
        self.temp = 1.0  # the temperature param
        self.best_win_ratio = 0.0
        self.train_step = 10
        self.n_games=100
        self.n_playout = 150
        self.eval_is_shown=1

        print(data)
        print("self.check_freq, self.epochs, self.batch_size, self.train_step, self.n_games, self.n_playout, self.is_shown")
        print(self.check_freq, self.epochs, self.batch_size, self.train_step, self.n_games, self.n_playout, self.eval_is_shown)

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            self.data_buffer.extend(play_data)
        return(len(play_data))

    def policy_update(self, epoch):
        """update the policy-value net"""
        loss = 0.0
        for _ in range(self.train_step):
            mini_batch = random.sample(self.data_buffer, self.batch_size)
            loss += self.player.train(mini_batch)
        loss /= self.train_step

        # Tensorboard
        if use_tensorboard:
            writer.add_scalar("Loss/train", loss, epoch)
        return loss
        

    def policy_evaluate(self, n_games=10, n_playout=1000, is_shown=0):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_player = self.player
        old_eps = current_player.eps
        current_player.eps = 0
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=n_playout)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=is_shown)
            win_cnt[winner] += 1
        current_player.eps = old_eps*1.2
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print(f"num_playouts:{n_games}, win: {win_cnt[1]}, lose: {win_cnt[2]}, tie:{win_cnt[-1]}")
        return win_ratio

    def run(self, check_pt_suffix:str):
        """run the training pipeline"""
        try:
            print("Train Started...")
            for i in tqdm(range(self.epochs)):
                move_count = self.collect_selfplay_data()

                # Becomes more greedy
                if self.player.eps > self.player.eps_threshold:
                    self.player.eps **= self.player.eps_decay
                    if self.player.eps > 0.9:
                        self.player.eps = 0.9
                else:
                    self.player.eps = self.player.eps_threshold
                
                if use_tensorboard:
                    writer.add_scalar("eps/Train", self.player.eps,i)
                if len(self.data_buffer) > self.batch_size:
                    loss = self.policy_update(i)
                
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print(f"\ncurrent self-play batch: {i+1}")
                    win_ratio = self.policy_evaluate(n_games=self.n_games, n_playout = self.n_playout, is_shown=self.eval_is_shown)
                    if use_tensorboard:
                        writer.add_scalar("win_rate/Validate",win_ratio,i)
                    self.player.save_model(f'./model/current_dqn_conv_'+check_pt_suffix+f'.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.player.save_model(f'./model/best_dqn_conv_'+check_pt_suffix+f'.model')
            print('Train Completed!')
            if use_tensorboard:
                writer.flush()
        except KeyboardInterrupt:
            print('\n\rquit')

use_tensorboard = True
if __name__ == '__main__':
    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
    training_pipeline = TrainPipeline()
    time_stamp = datetime.datetime.now().__str__()[2:-7]
    print(time_stamp)
    custom_name = "6by6_single_q"
    training_pipeline.run(re.sub(r'[^\w\-_\. ]', '_',(custom_name+"_"+time_stamp)))
