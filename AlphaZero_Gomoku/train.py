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
        self.daemon=True
    def run(self):
        try:
            winner = self.game.start_play(self.curr_mcts,
                                            self.pure_mcts,
                                            start_player=self.round % 2,
                                            is_shown=0)
            with winner_cnt_lock:
                self.winner_cnt[winner]+=1
        except KeyboardInterrupt:
            print(f"Terminating round{self.round}")

from torch.utils.tensorboard import SummaryWriter

class TrainPipeline():
    def __init__(self, uuid = "0000", resume = False, force_cpu = False):
        # load data from json file
        
        self.uuid = uuid
        self.io_dir = f"../models/{str(uuid)}/"
        self.output_json_path = self.io_dir+f"output.json"
        output_num=0
        while os.path.exists(self.output_json_path):
            output_num = output_num+1
            self.output_json_path = self.io_dir+f"output{output_num}.json"
        model_config = self.io_dir + f"model.json"
        train_config = self.io_dir + f"train.json"
        with open(model_config, encoding='utf-8') as f:
            model_config = json.loads(f.read())
        with open(train_config, encoding='utf-8') as f:
            train_config = json.loads(f.read())

        # params of the board and the game
        self.board_width = model_config["board"]["board_width"]
        self.board_height = model_config["board"]["board_height"]
        self.n_in_row = model_config["board"]["n_in_row"]
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # training params
        self.lr = train_config["hyperparameters"]["lr"]
        self.buffer_size = train_config["hyperparameters"]["buffer_size"]
        self.batch_size = train_config["hyperparameters"]["batch_size"]
        self.epochs = train_config["hyperparameters"]["epochs"]
        self.eval_rounds = train_config["testparameters"]["eval_rounds"]
        self.model_playout = train_config["testparameters"]["model_playout"]  # num of simulations for each move

        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = train_config["testparameters"]["mcts_playout"]
        self.check_freq = train_config["testparameters"]["check_freq"]
        
        self.data_buffer = deque(maxlen=self.buffer_size)
        
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.c_puct = 5
        self.play_batch_size = 1
        self.kl_targ = 0.02
        self.best_win_ratio = 0.0
        
        
        model_file_path = f"../models/{str(self.uuid)}/curr.model"
        if resume and os.path.exists(model_file_path):
            print(f"Loading checkpoint from: {str(self.uuid)}")
        else:
            print("Training new checkpoints.", end = " ")
            if os.path.exists(model_file_path):
                print("Overriding "+model_file_path, end = "")
            model_file_path = None
            print(flush=True)
        if force_cpu:
            print("Forced to use CPU only")

        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_config["nn_type"], model_config["layers"], model_file = model_file_path,force_cpu=force_cpu)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.model_playout,
                                      is_selfplay=1)

        self.writer = SummaryWriter()
        # {"train_progression":[
        #       [0epoch, 1time, 2loss, 3entropy, 4D_kl],
        #        ... ,
        #   ],
        #  "win_rates":[
        #       [epoch, win_rate]
        #   ]
        #  }
        self.records = {"start":"","train_progression":[],"win_rates":[],"end":""}
        json.dump(self.records,open(self.output_json_path,"w"))
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
        self.writer.add_scalar("KL Divergence", kl, self.step)
        self.writer.add_scalar("Loss", loss, self.step)
        self.writer.add_scalar("Entropy", entropy, self.step)
        self.step += 1
        return loss, entropy, kl

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.model_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        threads = []
        for ii in range(n_games):
            new_thread = Eval_Thread(self, current_mcts_player, pure_mcts_player, ii, win_cnt)
            threads.append(new_thread)
            new_thread.start()
        for t in threads:
            try:
                t.join()
            except KeyboardInterrupt:
                print("Ignoring a thread")
                raise KeyboardInterrupt
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                n_games, win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:        
            timestamp = re.sub(r'[^\w\-_\. ]', '_', datetime.datetime.now().__str__()[2:-7])
            self.records["start"]=timestamp
            start = time.time()
            for ii in range(self.epochs):
                self.collect_selfplay_data(self.play_batch_size)
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy, kl = self.policy_update(ii)
                    self.records["train_progression"].append([int(ii), # epoch
                                                            float(round(time.time()-start,2)), # elapsed time
                                                            float(round(loss,5)),
                                                            float(round(entropy,5)),
                                                            float(round(kl,5))])
                    json.dump(self.records,open(self.output_json_path,"w"))
                # check the performance of the current model,
                # and save the model params
                if (ii+1) % self.check_freq == 0:
                    win_ratio = self.policy_evaluate(self.eval_rounds)
                    self.records["win_rates"].append([ii,float(round(win_ratio,2))])
                    self.policy_value_net.save_model(f"../models/"
                                                    f"{self.uuid}/"
                                                    f"curr.model")
                    if win_ratio > self.best_win_ratio:
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(f"../models/"
                                                        f"{self.uuid}/"
                                                        f"best.model")
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                json.dump(self.records,open(self.output_json_path,"w"))
            self.policy_evaluate(self.eval_rounds)
            # Save at the end of training             
            self.policy_value_net.save_model(f"../models/"
                                            f"{self.uuid}/"
                                            f"curr.model")
            self.writer.close()
        except KeyboardInterrupt:
            print('\n\rquit')
            # Save at the end of training             
            self.policy_value_net.save_model(f"../models/"
                                            f"{self.uuid}/"
                                            f"curr.model")
        timestamp = re.sub(r'[^\w\-_\. ]', '_', datetime.datetime.now().__str__()[2:-7])
        self.records["end"] = timestamp
        json.dump(self.records,open(self.output_json_path,"w"))

            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-u","--uuid", help="UUID is used for reading model parameters and saving, loading models")
    parser.add_argument("-r","--resume", action = "store_true" , help="Resume from saved checkpoint", default=False)
    parser.add_argument("-c","--cpu", action="store_true",help="Force to run on CPU, without cuda", default=False)
    args = parser.parse_args()
    winner_cnt_lock = threading.Lock()

    test_uuid = args.uuid
    test_resume = args.resume
    test_force_cpu = args.cpu
    training_pipeline = TrainPipeline(test_uuid, test_resume, test_force_cpu)
    training_pipeline.run()
