# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

import json
import random
import datetime
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from nn_architecture import PolicyValueNet

import os
import argparse
import time
import re
import threading
import copy

class Eval_Thread(threading.Thread):
    def __init__(self, train_pipeline, curr_mcts, pure_mcts, round_num, winner_cnt):
        threading.Thread.__init__(self)
        self.train_pipeline = train_pipeline
        self.game = copy.deepcopy(train_pipeline.game)
        self.curr_mcts = copy.deepcopy(curr_mcts)
        self.pure_mcts = copy.deepcopy(pure_mcts)
        self.round =round_num
        self.winner_cnt = winner_cnt
    def run(self):
        winner = self.game.start_play(self.curr_mcts,
                                          self.pure_mcts,
                                          start_player=self.round % 2,
                                          is_shown=0)
        with winner_cnt_lock:
            self.winner_cnt[winner]+=1

from torch.utils.tensorboard import SummaryWriter

class TrainPipeline():
    def __init__(self, uuid = "0000", resume = False):
        # load data from json file
        
        self.uuid = uuid
        data = f"../models/{str(uuid)}/model.json"
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
        
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.play_batch_size = 1
        self.kl_targ = 0.02
        self.check_freq = 100
        self.best_win_ratio = 0.0
        self.eval_rounds = 50
        
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000

        if resume:
            model_file = f"../models/{str(self.uuid)}/curr.model"
            print(f"Loading checkpoint from: {str(self.uuid)}")
        else:
            model_file = None
            print("Training new checkpoints.", end = " ")
            if os.path.exists(f"../models/{str(self.uuid)}/curr.model"):
                print("Overriding "+f"../models/{str(self.uuid)}/best.model", end = "")
            print(flush=True)
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, data["nn_information"], model_file = model_file)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

        self.writer = SummaryWriter()
        self.step = 0

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self, epoch):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = np.array([data[0] for data in mini_batch])
        mcts_probs_batch = np.array([data[1] for data in mini_batch])
        winner_batch = np.array([data[2] for data in mini_batch])
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(5):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.lr*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        # print(("kl:{:.5f},"
        #        "lr_multiplier:{:.3f},"
        #        "loss:{},"
        #        "entropy:{},"
        #        "explained_var_old:{:.3f},"
        #        "explained_var_new:{:.3f}"
        #        ).format(kl,
        #                 self.lr_multiplier,
        #                 loss,
        #                 entropy,
        #                 explained_var_old,
        #                 explained_var_new))
        print(f"epoch {epoch:05d} | loss: {loss}", end = "", flush=True)
        self.writer.add_scalar("KL Divergence", kl, self.step)
        self.writer.add_scalar("Loss", loss, self.step)
        self.writer.add_scalar("Entropy", entropy, self.step)
        self.step += 1
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        threads = []
        for ii in range(n_games):
            new_thread = Eval_Thread(self, current_mcts_player, pure_mcts_player, ii, win_cnt)
            threads.append(new_thread)
            new_thread.start()
        for t in threads:
            t.join()
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                n_games, win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:        
            timestamp = re.sub(r'[^\w\-_\. ]', '_', datetime.datetime.now().__str__()[2:-7])
            start = time.time()
            print(self.uuid, timestamp)
            for ii in range(self.epochs):
                print(f"epoch {ii:05d} | elapsed time: {time.time()-start:.2f}",end = "",flush=True)

                self.collect_selfplay_data(self.play_batch_size)
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update(ii)
                # check the performance of the current model,
                # and save the model params
                if (ii+1) % self.check_freq == 0:
                    # print("\ncurrent self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate(self.eval_rounds)
                    self.policy_value_net.save_model(f"../models/"
                                                    f"{self.uuid}/"
                                                    f"curr.model")
                    if win_ratio > self.best_win_ratio:
                        #print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model(f"../models/"
                                                        f"{self.uuid}/"
                                                        f"best.model")
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
            self.policy_evaluate(self.eval_rounds)
            # Save at the end of training             
            self.policy_value_net.save_model(f"../models/"
                                            f"{self.uuid}/"
                                            f"curr.model")
            self.writer.close()
        except KeyboardInterrupt:
            print('\n\rquit')
        timestamp = re.sub(r'[^\w\-_\. ]', '_', datetime.datetime.now().__str__()[2:-7])
        print("Train finished" f"{self.uuid} {timestamp}")
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-u","--uuid", help="UUID is used for reading model parameters and saving, loading models")
    parser.add_argument("-r","--resume", action = "store_true" , help="Resume from saved checkpoint", default=False)
    args = parser.parse_args()
    winner_cnt_lock = threading.Lock()

    uuid=args.uuid
    # comment this line out before deploying
    uuid = "1aaa41fa-526e-47c6-916c-07906127df3c"
    training_pipeline = TrainPipeline(uuid, args.resume)
    training_pipeline.run()
