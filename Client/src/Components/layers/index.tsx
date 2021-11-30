import { isActivationFunction } from '../../lib/util';
import {
  ActivationFunction,
  ConvLayer,
  GraphConvLayer,
  Layer,
  LayerType
} from '../../model/types';
import ActivationBlock from './activationBlock';
import BatchNorm from './batchNorm';
import Conv from './conv';
import GraphConv from './graphConv';

const createLayerElement = (
  layerType: LayerType | ActivationFunction,
  layerProps: ConvLayer | GraphConvLayer | Layer,
  key?: any,
  onClick?: () => void,
  onClose?: () => void,
  onModify?: (newLayer: ConvLayer | GraphConvLayer | Layer) => () => void
) => {
  if (isActivationFunction(layerType)) {
    return (
      <ActivationBlock
        layerType={layerType}
        onClick={onClick}
        onClose={onClose}
        key={key}
      />
    );
  }
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

export { createLayerElement };
