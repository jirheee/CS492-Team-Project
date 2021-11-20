import Model from './model';
import { Board, HyperParameters } from './types';

class BattleInfo {
  public board: Board;
  public player1: string;
  public player2: string;

  constructor(board: Board, hyperParameters: HyperParameters, model: Model) {
    this.board = board;
    this.player1 = 'player1 model uuid';
    this.player2 = 'player2 model uuid';
  }
}

export default BattleInfo;
