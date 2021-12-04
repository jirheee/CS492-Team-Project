export interface HyperParameters {
  lr: number;
  buffer_size: number;
  batch_size: number;
  epochs: number;
}

export interface Board {
  board_width: number;
  board_height: number;
  n_in_row: number;
}

export enum LayerType {
  Conv = 'Conv',
  BatchNorm = 'BatchNorm',
  GCNConv = 'GCNConv',
  SGConv = 'SGConv',
  GATConv = 'GATConv',
  SAGEConv = 'SAGEConv'
}

export enum ActivationFunction {
  ReLU = 'ReLU',
  Sigmoid = 'Sigmoid',
  Tanh = 'Tanh',
  LeakyReLU = 'LeakyReLU'
}

export enum NNType {
  CNN = 'CNN',
  GNN = 'GNN'
}

export enum BiasInfo {
  False = 'False',
  True = 'True'
}

export interface Layer {
  layer_name: LayerType | ActivationFunction;
}

export interface GraphConvLayer extends Layer {
  channels: number;
  bias: BiasInfo;
}

export interface ConvLayer extends GraphConvLayer {
  kernel_size: number;
  stride: number;
  padding: number;
}
