import React from 'react';
import { Flex } from '@chakra-ui/react';
import { ConvLayer, GraphConvLayer, Layer } from '../model/types';
import { createLayerElement } from './layers';

interface NNBoardProps {
  layers: (ConvLayer | GraphConvLayer | Layer)[];
  setLayers?: React.Dispatch<
    React.SetStateAction<(ConvLayer | GraphConvLayer | Layer)[]>
  >;
}

const NNBoard = ({ layers, setLayers }: NNBoardProps) => {
  return (
    <Flex
      h="full"
      minH="400px"
      w="full"
      border="solid"
      borderColor="gray.200"
      borderRadius={30}
      overflow="hidden"
      alignItems="center"
      justifyContent="center"
    >
      {layers.map((layerProps, i) =>
        createLayerElement(
          layerProps.layer_name,
          layerProps,
          i,
          undefined,
          () => {
            setLayers &&
              setLayers(layerArr => {
                const newLayerArr = [...layerArr];
                newLayerArr.splice(i, 1);
                return newLayerArr;
              });
          },
          newLayer => () => {
            setLayers &&
              setLayers(layerArr => {
                const newLayerArr = [...layerArr];
                newLayerArr.splice(i, 1, newLayer);
                return newLayerArr;
              });
          }
        )
      )}
    </Flex>
  );
};

export default NNBoard;
