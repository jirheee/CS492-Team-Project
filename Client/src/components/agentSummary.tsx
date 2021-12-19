import { Flex, Avatar, Text, Grid, Center } from '@chakra-ui/react';
import Model from '../model/model';
import { HyperParameters } from '../model/types';
import { Agent, TrainStatus } from '../pages/agentManagementPage';

interface AgentSummaryProps {
  model?: Model;
  trainInfo?: {
    hyperparameters: HyperParameters;
    trainStatus: TrainStatus;
  };
  agent?: Agent;
}

const AgentSummary = ({ model, trainInfo, agent }: AgentSummaryProps) => {
  return (
    <Grid templateColumns="repeat(5, 1fr)">
      {/* Avatar Picture and Name */}
      <Flex flexDir="column" justifyContent="center">
        <Center>
          <Avatar size="xl" />
        </Center>
        <Text textAlign="center">{model?.name}</Text>
      </Flex>

      {/* Network Information */}
      <Flex flexDir="column">
        <Text fontWeight="bold">Network Info</Text>
        <Text>Network Type: {model?.nn_type}</Text>
        <Text>Number on Parameters</Text>
      </Flex>

      {/* information about board */}
      <Flex flexDir="column">
        <Text fontWeight="bold">Board Info</Text>
        <Text>Board Width: {model?.board.board_width ?? 'NaN'}</Text>
        <Text>N in Row: {model?.board.n_in_row ?? 'NaN'}</Text>
      </Flex>

      {/* training information */}
      <Flex flexDir="column">
        <Text fontWeight="bold">Training Info</Text>
        <Text>Learning Rate: {trainInfo?.hyperparameters.lr ?? 'NaN'}</Text>
        <Text>
          Buffer Size: {trainInfo?.hyperparameters.buffer_size ?? 'NaN'}
        </Text>
        <Text>
          Batch Size: {trainInfo?.hyperparameters.batch_size ?? 'NaN'}
        </Text>
        <Text>Epochs: {trainInfo?.hyperparameters.epochs ?? 'NaN'}</Text>
      </Flex>

      {/* battle information */}
      <Flex flexDir="column">
        <Text fontWeight="bold">Battle Info</Text>
        <Text>
          {agent?.win}W {agent?.lose}L
        </Text>
      </Flex>
    </Grid>
  );
};

export default AgentSummary;
