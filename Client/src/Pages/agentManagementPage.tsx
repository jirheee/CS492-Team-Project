import { useState } from 'react';
import { Container, Flex } from '@chakra-ui/react';
import NNBoard from '../components/nnBoard';
import { ConvLayer, GraphConvLayer, Layer } from '../model/types';
import AgentSummary from '../components/agentSummary';

interface AgentManagementPageProps {
  agentUUID: string;
}

const AgentManagementPage = ({ agentUUID }: AgentManagementPageProps) => {
  const [layers, setLayers] = useState<(ConvLayer | GraphConvLayer | Layer)[]>(
    []
  );
  const [trainHistory, setTrainHistory] = useState<number[]>([]);

  return (
    <Flex flexDir="column" justifyContent="center" backgroundColor="gray.50">
      <Container maxW={1200} backgroundColor="white" h="full">
        <AgentSummary />
        <NNBoard layers={layers} />
      </Container>
    </Flex>
  );
};

export default AgentManagementPage;
