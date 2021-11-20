import { LayerType } from '../model/types';

const getBlockColor = (layerType: LayerType) => {
  switch (layerType) {
    case LayerType.BatchNorm:
      return 'yellow.100';
    case LayerType.Conv:
      return 'blue.300';
    case LayerType.GATConv:
      return 'orange.50';
    case LayerType.GCNConv:
      return 'orange.100';
    case LayerType.GINConv:
      return 'orange.200';
    case LayerType.SAGEConv:
      return 'orange.300';
    case LayerType.SGConv:
      return 'orange.400';
  }
};

const isInt = value => {
  return /^\+?(0|[1-9]\d*)$/.test(value);
};

const getGnnAvailableLayerTypes = () => [
  LayerType.BatchNorm,
  LayerType.GATConv,
  LayerType.GCNConv,
  LayerType.GINConv,
  LayerType.SAGEConv,
  LayerType.SGConv
];

const getCnnAvailableLayerTypes = () => [LayerType.BatchNorm, LayerType.Conv];

export {
  getBlockColor,
  isInt,
  getGnnAvailableLayerTypes,
  getCnnAvailableLayerTypes
};
