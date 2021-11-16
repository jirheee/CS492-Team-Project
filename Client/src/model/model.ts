import { ActivationFunction, Board, Layer, NNType } from './types';

class Model {
  public board: Board = { board_height: 6, board_width: 6, n_in_row: 4 };
  public nn_type: NNType = NNType.CNN;
  public n_layers: number = 0;
  public layers: Layer[] = [];
  public activation: ActivationFunction = ActivationFunction.ReLU;
}

export default Model;
