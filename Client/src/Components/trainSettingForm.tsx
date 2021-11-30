import {
  Grid,
  FormControl,
  Select,
  FormLabel,
  Input,
  Button
} from '@chakra-ui/react';

const TrainSettingForm = () => {
  const handleSubmit = () => {};
  return (
    <Grid
      as="form"
      marginLeft="auto"
      m={5}
      minW="200px"
      h="full"
      onSubmit={handleSubmit}
    >
      <FormControl isRequired>
        <FormLabel>Learning Rate</FormLabel>
        <Input placeholder="Type in your Agent's Nickname" />
      </FormControl>
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
      <Button mt={4} colorScheme="teal" type="submit">
        Create Agent
      </Button>
    </Grid>
  );
};
