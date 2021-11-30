import { Grid, Flex, Text } from '@chakra-ui/react';
import {
  getCnnAvailableLayerTypes,
  getGnnAvailableLayerTypes,
  getActivationFunctions
} from '../lib/util';
import { ConvLayer, GraphConvLayer, NNType } from '../model/types';
import { createLayerElement } from './layers';

interface NNItemShelfProps {
  nnType: NNType;
  onClick?: (layerType) => () => void;
}

const NNItemShelf = ({ nnType, onClick }: NNItemShelfProps) => {
  const trainableBlockNumber = nnType === NNType.CNN ? 2 : 5;
  const activationBlockNumber = 4;
  return (
    <Flex>
      <Flex
        marginTop="auto"
        border="solid"
        borderRadius={30}
        borderColor="gray.200"
        h="full"
        w={`${
          (trainableBlockNumber /
            (trainableBlockNumber + activationBlockNumber)) *
          100
        }%`}
        mt={5}
        marginRight={3}
        flexDir="column"
        overflow="hidden"
      >
        <Text p={3} w="full" backgroundColor="gray.100" fontWeight="bold">
          CNN Blocks
        </Text>
        <Grid
          w="full"
          p={5}
          templateColumns={`repeat(${trainableBlockNumber}, 1fr)`}
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
      </Flex>
      <Flex
        marginTop="auto"
        border="solid"
        borderRadius={30}
        borderColor="gray.200"
        h="full"
        w={`${
          (activationBlockNumber /
            (trainableBlockNumber + activationBlockNumber)) *
          100
        }%`}
        mt={5}
        marginRight={3}
        flexDir="column"
        overflow="hidden"
      >
        <Text p={3} w="full" backgroundColor="gray.100" fontWeight="bold">
          Activation Blocks
        </Text>
        <Grid
          w="full"
          templateColumns="repeat(4, 1fr)"
          alignContent="center"
          p={5}
        >
          {getActivationFunctions().map(layerProps =>
            createLayerElement(
              layerProps.layer_name,
              layerProps,
              layerProps.layer_name,
              onClick && onClick(layerProps.layer_name)
            )
          )}
        </Grid>
      </Flex>
    </Flex>
  );
};

export default NNItemShelf;
