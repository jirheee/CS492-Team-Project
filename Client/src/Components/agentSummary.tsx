import { Flex, Avatar, Text, Grid, Center } from '@chakra-ui/react';

const AgentSummary = () => {
  return (
    <Grid templateColumns="repeat(5, 1fr)">
      {/* Avatar Picture and Name */}
      <Flex flexDir="column" justifyContent="center">
        <Center>
          <Avatar size="xl" />
        </Center>
        <Text textAlign="center">Name</Text>
      </Flex>

      {/* Network Information */}
      <Flex flexDir="column">
        <Text>Network Info</Text>
        <Text>Network Type</Text>
        <Text>Number on Parameters</Text>
      </Flex>

      {/* information about board */}
      <Flex flexDir="column">
        <Text>Board Info</Text>
        <Text>Board Width</Text>
        <Text>N in Row</Text>
      </Flex>

      {/* training information */}
      <Flex flexDir="column">
        <Text>Training Info</Text>
        <Text>Learning Rate</Text>
        <Text>Buffer Size</Text>
        <Text>Batch Size</Text>
        <Text>Epochs</Text>
        <Text>Training Time</Text>
      </Flex>

      {/* battle information */}
      <Flex flexDir="column">
        <Text>Battle Info</Text>
        <Text>Ranking</Text>
        <Text>10W 10L (50%)</Text>
      </Flex>
    </Grid>
  );
};

export default AgentSummary;
