import { Flex } from '@chakra-ui/react';
import BatchNorm from '../components/layers/batchNorm';
import Conv from '../components/layers/conv';
import GraphConv from '../components/layers/graphConv';
import { LayerType } from '../model/types';

const AgentCreatePage = () => {
  return (
    <Flex h="100vh">
      <Conv index={0} />
      <GraphConv layerType={LayerType.GATConv} index={1} />
      <BatchNorm index={2} />
    </Flex>
  );
};

export default AgentCreatePage;
