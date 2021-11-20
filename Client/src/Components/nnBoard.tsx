import React, { useState } from 'react';
import {
  Flex,
  Select,
  FormControl,
  Button,
  FormLabel,
  Input,
  Grid
} from '@chakra-ui/react';
import {
  ActivationFunction,
  ConvLayer,
  GraphConvLayer,
  Layer,
  LayerType,
  NNType
} from '../model/types';
import Conv from './layers/conv';
import GraphConv from './layers/graphConv';
import BatchNorm from './layers/batchNorm';
import {
  createLayer,
  getCnnAvailableLayerTypes,
  getGnnAvailableLayerTypes,
  isInt
} from '../lib/util';
import Request from '../lib/api/request';

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

const NNBoard = () => {
  const [nnType, setNNtype] = useState<NNType>(NNType.CNN);
  const [layers, setLayers] = useState<(ConvLayer | GraphConvLayer | Layer)[]>(
    []
  );
  const [isInvalidBoardWidth, setInvalidBoardWidth] = useState<boolean>(false);
  const [isInvalidNinRow, setInvalidNinRow] = useState<boolean>(false);

  const handleSubmit = async event => {
    console.log(event);
    event.preventDefault();
    const { target } = event;
    console.log(
      JSON.stringify({
        board: {
          board_width: +target[1].value,
          board_height: +target[1].value,
          n_in_row: +target[2].value
        },
        nn_type: target[0].value,
        layers,
        activ_func: 'ReLU'
      })
    );
    // Request.post('/create/agent', JSON.stringify(layers));
    console.log(target);
  };

  const validateBoardWidth = e => {
    const { value } = e.target;
    if (!isInt(value)) {
      setInvalidBoardWidth(true);
    } else {
      setInvalidBoardWidth(false);
    }
  };

  const validateNinRow = e => {
    const { value } = e.target;
    if (!isInt(value)) {
      setInvalidNinRow(true);
    } else {
      setInvalidNinRow(false);
    }
  };

  const handleNNTypeChange = e => {
    const newNNtype = e.target.value;
    if (newNNtype !== nnType) {
      setNNtype(newNNtype);
      setLayers([]);
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
                setLayers(layerArr => {
                  const newLayerArr = [...layerArr];
                  newLayerArr.splice(i, 1);
                  return newLayerArr;
                });
              },
              newLayer => () => {
                setLayers(layerArr => {
                  const newLayerArr = [...layerArr];
                  newLayerArr.splice(i, 1, newLayer);
                  return newLayerArr;
                });
              }
            )
          )}
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
            <Select onChange={handleNNTypeChange} defaultValue={NNType.CNN}>
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
          <FormControl isRequired>
            <FormLabel>Activation Function</FormLabel>
            <Select>
              <option value={ActivationFunction.ReLU}>
                {ActivationFunction.ReLU}
              </option>
              <option value={ActivationFunction.LeakyReLU}>
                {ActivationFunction.LeakyReLU}
              </option>
              <option value={ActivationFunction.Sigmoid}>
                {ActivationFunction.Sigmoid}
              </option>
              <option value={ActivationFunction.Tanh}>
                {ActivationFunction.Tanh}
              </option>
            </Select>
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
                () => {
                  setLayers(layerArr => [
                    ...layerArr,
                    createLayer(layerProps.layer_name) as ConvLayer
                  ]);
                }
              )
            )
          : getGnnAvailableLayerTypes().map(layerProps =>
              createLayerElement(
                layerProps.layer_name,
                layerProps as GraphConvLayer,
                layerProps.layer_name,
                () => {
                  setLayers(layerArr => [
                    ...layerArr,
                    createLayer(layerProps.layer_name) as GraphConvLayer
                  ]);
                }
              )
            )}
      </Grid>
    </Flex>
  );
};

export default NNBoard;
