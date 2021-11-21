import { Grid } from '@chakra-ui/react';
import {
  getCnnAvailableLayerTypes,
  getGnnAvailableLayerTypes
} from '../lib/util';
import { ConvLayer, GraphConvLayer, NNType } from '../model/types';
import { createLayerElement } from './layers';

interface NNItemShelfProps {
  nnType: NNType;
  onClick?: (layerType) => () => void;
}

const NNItemShelf = ({ nnType, onClick }: NNItemShelfProps) => {
  return (
    <Grid
      marginTop="auto"
      border="solid"
      borderRadius={30}
      borderColor="gray.200"
      h="full"
      mt={5}
      p={5}
      templateColumns="repeat(12, 1fr)"
      alignContent="center"
    >
      {nnType === NNType.CNN
        ? getCnnAvailableLayerTypes().map(layerProps =>
            createLayerElement(
              layerProps.layer_name,
              layerProps as ConvLayer,
              layerProps.layer_name,
              onClick && onClick(layerProps.layer_name)
            )
          )
        : getGnnAvailableLayerTypes().map(layerProps =>
            createLayerElement(
              layerProps.layer_name,
              layerProps as GraphConvLayer,
              layerProps.layer_name,
              onClick && onClick(layerProps.layer_name)
            )
          )}
    </Grid>
  );
};

export default NNItemShelf;
