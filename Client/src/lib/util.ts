import { LayerType } from '../model/types';

const getBlockColor = (layerType: LayerType) => {
  switch (layerType) {
    case LayerType.BatchNorm:
      return 'yellow.100';
    case LayerType.Conv:
      return 'blue.300';
    default:
      return 'orange.300';
  }
};

export { getBlockColor };
