import { ConvLayer, GraphConvLayer, Layer, LayerType } from '../../model/types';
import BatchNorm from './batchNorm';
import Conv from './conv';
import GraphConv from './graphConv';

const createLayerElement = (
  layerType: LayerType,
  layerProps: ConvLayer | GraphConvLayer | Layer,
  key?: any,
  onClick?: () => void,
  onClose?: () => void,
  onModify?: (newLayer: ConvLayer | GraphConvLayer | Layer) => () => void
) => {
  switch (layerType) {
    case LayerType.BatchNorm:
      return (
        <BatchNorm
          onClick={onClick}
          onClose={onClose}
          onModify={onModify}
          key={key}
        />
      );
    case LayerType.Conv:
      return (
        <Conv
          layerProps={layerProps as ConvLayer}
          onClick={onClick}
          onClose={onClose}
          onModify={onModify}
          key={key}
        />
      );
    default:
      return (
        <GraphConv
          layerType={layerType}
          layerProps={layerProps as GraphConvLayer}
          onClick={onClick}
          onClose={onClose}
          onModify={onModify}
          key={key}
        />
      );
  }
};

export { BatchNorm, Conv, GraphConv, createLayerElement };
