import React, { useState } from 'react';
import {
  Flex,
  Select,
  FormControl,
  Button,
  FormLabel,
  Input,
  Grid,
  Box
} from '@chakra-ui/react';
import { Layer, LayerType, NNType } from '../model/types';
import Conv from './layers/conv';
import GraphConv from './layers/graphConv';
import BatchNorm from './layers/batchNorm';
import {
  getCnnAvailableLayerTypes,
  getGnnAvailableLayerTypes,
  isInt
} from '../lib/util';

const createLayer = (layerType: LayerType): Layer => {
  switch (layerType) {
    case LayerType.BatchNorm:
      return { layer_name: LayerType.BatchNorm };
    case LayerType.Conv:
      return { layer_name: LayerType.Conv };
    default:
      return { layer_name: layerType };
  }
};

const createLayerElement = (layerType: LayerType, isEditable: boolean) => {
  switch (layerType) {
    case LayerType.BatchNorm:
      return <BatchNorm isEditable={isEditable} />;
    case LayerType.Conv:
      return <Conv isEditable={isEditable} />;
    default:
      return <GraphConv layerType={layerType} isEditable={isEditable} />;
  }
};

const NNBoard = () => {
  const [nnType, setNNtype] = useState<NNType>(NNType.CNN);
  const [layers, setLayers] = useState<Layer[]>([]);
  const [isInvalidBoardWidth, setInvalidBoardWidth] = useState<boolean>(false);
  const [isInvalidNinRow, setInvalidNinRow] = useState<boolean>(false);

  const handleSubmit = async event => {
    console.log(event);
    event.preventDefault();
    const { target } = event;
    console.log(target);
  };

  const validateBoardWidth = e => {
    const { value } = e.target;
    console.log(isInt(value));
    if (!isInt(value)) {
      setInvalidBoardWidth(true);
    } else {
      console.log(isInvalidBoardWidth);
      setInvalidBoardWidth(false);
    }
  };

  const validateNinRow = e => {
    const { value } = e.target;
    console.log(isInt(value));
    if (!isInt(value)) {
      setInvalidNinRow(true);
    } else {
      console.log(isInvalidBoardWidth);
      setInvalidNinRow(false);
    }
  };

  return (
    <Flex w="full" h="full" p={5} flexDir="column">
      <Flex w="full" minH="70%">
        <Flex
          h="full"
          w="full"
          border="solid"
          borderColor="gray.200"
          borderRadius={30}
          overflow="hidden"
          flexDir="column"
        >
          <Flex alignContent="center" justifyContent="center" h="full">
            {layers.map(({ layer_name }) =>
              createLayerElement(layer_name, true)
            )}
          </Flex>
        </Flex>
        <Grid
          as="form"
          marginLeft="auto"
          m={5}
          minW="200px"
          h="full"
          onSubmit={handleSubmit}
        >
          <FormControl isRequired>
            <FormLabel>Network Type</FormLabel>
            <Select
              placeholder="Select Network Type"
              onChange={e => setNNtype(e.target.value as NNType)}
            >
              <option value={NNType.CNN}>{NNType.CNN}</option>
              <option value={NNType.GNN}>{NNType.GNN}</option>
            </Select>
          </FormControl>
          <FormControl isRequired isInvalid={isInvalidBoardWidth}>
            <FormLabel>Board Width</FormLabel>
            <Input defaultValue={6} onChange={validateBoardWidth} />
          </FormControl>
          <FormControl isRequired isInvalid={isInvalidNinRow}>
            <FormLabel>N in Row</FormLabel>
            <Input defaultValue={4} onChange={validateNinRow} />
          </FormControl>
          <Button mt={4} colorScheme="teal" type="submit">
            Create Agent
          </Button>
        </Grid>
      </Flex>
      <Grid
        marginTop="auto"
        border="solid"
        borderRadius={30}
        borderColor="gray.200"
        p={5}
        templateColumns="repeat(10, 1fr)"
      >
        {nnType === NNType.CNN
          ? getCnnAvailableLayerTypes().map(layerType =>
              createLayerElement(layerType, false)
            )
          : getGnnAvailableLayerTypes().map(layerType =>
              createLayerElement(layerType, false)
            )}
      </Grid>
    </Flex>
  );
};

export default NNBoard;
