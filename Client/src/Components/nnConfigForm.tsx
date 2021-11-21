import { useState } from 'react';
import {
  Grid,
  FormControl,
  Select,
  FormLabel,
  Input,
  Button
} from '@chakra-ui/react';
import { ActivationFunction, NNType } from '../model/types';
import { isInt } from '../lib/util';

interface NNConfigFormProps {
  handleSubmit: (event) => void;
  handleNNTypeChange: (event) => void;
}

const NNConfigForm = ({
  handleSubmit,
  handleNNTypeChange
}: NNConfigFormProps) => {
  const [isInvalidBoardWidth, setInvalidBoardWidth] = useState<boolean>(false);
  const [isInvalidNinRow, setInvalidNinRow] = useState<boolean>(false);

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
  );
};

export default NNConfigForm;
