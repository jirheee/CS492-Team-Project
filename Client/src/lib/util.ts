import {
  ActivationFunction,
  BiasInfo,
  ConvLayer,
  GraphConvLayer,
  Layer,
  LayerType
} from '../model/types';

const getBlockColor = (layerType: LayerType | ActivationFunction) => {
  switch (layerType) {
    case LayerType.BatchNorm:
      return 'yellow.100';
    case LayerType.Conv:
      return 'blue.300';
    case LayerType.GATConv:
      return 'orange.50';
    case LayerType.GCNConv:
      return 'orange.100';
    case LayerType.SAGEConv:
      return 'orange.300';
    case LayerType.SGConv:
      return 'orange.400';
    default:
      return 'red.100';
  }
};

const isInt = value => {
  return /^\+?(0|[1-9]\d*)$/.test(value);
};

const isActivationFunction = (layerType: LayerType | ActivationFunction) => {
  return (
    layerType === ActivationFunction.LeakyReLU ||
    layerType === ActivationFunction.ReLU ||
    layerType === ActivationFunction.Tanh ||
    layerType === ActivationFunction.Sigmoid
  );
};

const createLayer = (
  layerType: LayerType | ActivationFunction
): Layer | GraphConvLayer | ConvLayer => {
  if (isActivationFunction(layerType)) {
    return { layer_name: layerType };
  }
  switch (layerType) {
    case LayerType.BatchNorm:
      return { layer_name: LayerType.BatchNorm };
    case LayerType.Conv:
      return {
        layer_name: LayerType.Conv,
        channels: 1,
        kernel_size: 1,
        stride: 1,
        padding: 1,
        bias: BiasInfo.True
      };
    default:
      return { layer_name: layerType, channels: 1, bias: BiasInfo.True };
  }
};

const createActivationLayer = (activationType: ActivationFunction) => {
  return { layer_name: activationType };
};

const getGnnAvailableLayerTypes = () =>
  [
    LayerType.GATConv,
    LayerType.GCNConv,
    LayerType.SAGEConv,
    LayerType.SGConv
  ].map(t => createLayer(t));

const getCnnAvailableLayerTypes = () =>
  [LayerType.BatchNorm, LayerType.Conv].map(t => createLayer(t));

const getActivationFunctions = () =>
  [
    ActivationFunction.ReLU,
    ActivationFunction.LeakyReLU,
    ActivationFunction.Sigmoid,
    ActivationFunction.Tanh
  ].map(t => createActivationLayer(t));

const capitalizeFirstChar = (str: string) => {
  return str.charAt(0).toLocaleUpperCase() + str.slice(1);
};

const lowercaseFirstChar = (str: string) => {
  return str.charAt(0).toLocaleLowerCase() + str.slice(1);
};

const snakeCaseToPascalCase = (str: string, seperator: string = '') => {
  return str.split('_').map(capitalizeFirstChar).join(seperator);
};

const pascalCaseToSnakeCase = (
  str: string,
  splitSeperator: string = '',
  joinSeperator = ''
) => {
  return str.split(splitSeperator).map(lowercaseFirstChar).join(joinSeperator);
};

const getMinimunvalueForField = field => {
  switch (field) {
    case 'kernel_size':
      return 1;
    case 'stride':
      return 0;
    case 'padding':
      return 0;
    case 'channels':
      return 1;
    default:
      console.error(field);
      return 0;
  }
};

export {
  getBlockColor,
  isInt,
  getGnnAvailableLayerTypes,
  getCnnAvailableLayerTypes,
  createLayer,
  snakeCaseToPascalCase,
  getMinimunvalueForField,
  lowercaseFirstChar,
  pascalCaseToSnakeCase,
  getActivationFunctions,
  isActivationFunction
};
