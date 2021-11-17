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
    | 'GINConv'
    | 'SAGEConv';
}

export interface Model {
  board: Board;
  nn_type: 'CNN' | 'GNN';
  layers: Required<Layer>[];
  active_func: 'ReLU' | 'Sigmoid' | 'Tanh' | 'LeakyReLU';
}

export interface HyperParameters {
  lr: number;
  buffer_size: number;
  batch_size: number;
  epochs: number;
}
