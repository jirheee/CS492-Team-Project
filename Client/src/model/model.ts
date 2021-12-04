import {
  ActivationFunction,
  BiasInfo,
  Board,
  ConvLayer,
  GraphConvLayer,
  Layer,
  LayerType,
  NNType
} from './types';

class Model {
  public name: string = '';
  public board: Board = { board_height: 6, board_width: 6, n_in_row: 4 };
  public nn_type: NNType = NNType.CNN;
  public layers: Layer[] = [];

  public addLayer(layerType: LayerType) {
    switch (layerType) {
      case LayerType.BatchNorm:
        const batchNorm: Layer = { layer_name: LayerType.BatchNorm };
        this.layers.push(batchNorm);
        break;
      case LayerType.Conv:
        const convLayer: ConvLayer = {
          layer_name: LayerType.Conv,
          kernel_size: 1,
          bias: BiasInfo.False,
          stride: 1,
          padding: 1,
          channels: 1
        };
        this.layers.push(convLayer);
        break;
      /** Graph Convolution Layers */
      default:
        const graphConvLayer: GraphConvLayer = {
          layer_name: layerType,
          bias: BiasInfo.False,
          channels: 1
        };
        this.layers.push(graphConvLayer);
        break;
    }
  }

  public removeLayer(index: number) {
    this.layers = this.layers.filter((_, i) => index !== i);
  }

  public modifyLayer(index: number, props: Record<string, unknown>) {
    this.layers[index] = { ...this.layers[index], ...props };
  }
}

export default Model;
