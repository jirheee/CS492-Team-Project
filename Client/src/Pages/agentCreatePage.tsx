import React, { useState } from 'react';
import { Flex } from '@chakra-ui/react';
import NNBoard from '../components/nnBoard';
import NNConfigForm from '../components/nnConfigForm';
import NNItemShelf from '../components/nnItemShelf';
import { createLayer } from '../lib/util';
import { ConvLayer, GraphConvLayer, Layer, NNType } from '../model/types';

const AgentCreatePage = () => {
  const [nnType, setNNtype] = useState<NNType>(NNType.CNN);
  const [layers, setLayers] = useState<(ConvLayer | GraphConvLayer | Layer)[]>(
    []
  );

  const handleSubmit = async event => {
    event.preventDefault();
    const { target } = event;
    console.log(
      JSON.stringify({
        name: target[0].value,
        board: {
          board_width: +target[2].value,
          board_height: +target[2].value,
          n_in_row: +target[3].value
        },
        nn_type: target[1].value,
        layers
      })
    );
    // Request.post('/create/agent', JSON.stringify(layers));
    console.log(target);
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
        <NNBoard layers={layers} setLayers={setLayers} />
        <NNConfigForm
          handleNNTypeChange={handleNNTypeChange}
          handleSubmit={handleSubmit}
        />
      </Flex>
      <NNItemShelf
        nnType={nnType}
        onClick={layerType => () => {
          setLayers(layerArr => [...layerArr, createLayer(layerType)]);
        }}
      />
    </Flex>
  );
};

export default AgentCreatePage;
