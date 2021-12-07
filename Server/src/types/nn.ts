export type AgentUUID = string;

export interface Board {
  board_width: number;
  board_height: number;
  n_in_row: number;
}

export interface Layer {
  layer_name:
    | 'Conv'
    | 'BatchNorm'
    | 'GCNConv'
    | 'SGConv'
    | 'GATConv'
    | 'SAGEConv'
    | 'ReLU'
    | 'Sigmoid'
    | 'Tanh'
    | 'LeakyReLU';
}

export interface Model {
  name: string;
  board: Board;
  nn_type: 'CNN' | 'GNN';
  layers: Required<Layer>[];
}

export interface HyperParameters {
  lr: number;
  buffer_size: number;
  batch_size: number;
  epochs: number;
}

export interface TrainHistory{
  start: string;
  train_progression: number[][];
  win_rates: number[][];
  end: string;
}

export enum TrainStatus {
  NOT_TRAINED = 'Not Trained',
  TRAINING = 'Training',
  TRAIN_FINISHED = 'Train Finished'
}
export interface TrainResponse {
  hyperparameters?: HyperParameters;
  trainStatus: TrainStatus;
}
