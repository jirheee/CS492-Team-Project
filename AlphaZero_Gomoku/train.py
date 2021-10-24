import json
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
from game import Board, Game
from nn_architecture import Conv
from rl_algorithm import DQNPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure


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
            self.nn_architecture = Conv(self.board_width, self.board_height)

        # Algorithm
        if data["rl_algorithm"] == 'DQN':
            self.player = DQNPlayer(self.nn_architecture, lr=self.lr)

        self.check_freq = 1000 # Checkpoint for evaluation
        self.temp = 1.0  # the temperature param
        self.best_win_ratio = 0.0

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            self.data_buffer.extend(play_data)

    def policy_update(self, epoch):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        loss = self.player.train(mini_batch)
        return loss
        

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_player = self.player
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=1000)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print(f"num_playouts:{n_games}, win: {win_cnt[1]}, lose: {win_cnt[2]}, tie:{win_cnt[-1]}")
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            print("Train Started...")
            for i in tqdm(range(self.epochs)):
                self.collect_selfplay_data()

                if len(self.data_buffer) > self.batch_size:
                    loss = self.policy_update(i)
                
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print(f"\ncurrent self-play batch: {i+1}")
                    win_ratio = self.policy_evaluate()
                    self.player.save_model(f'./model/current_dqn_conv.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.player.save_model(f'./model/best_dqn_conv.model')
                        if self.best_win_ratio == 1.0:
                            self.best_win_ratio = 0.0
            print('Train Completed!')
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
