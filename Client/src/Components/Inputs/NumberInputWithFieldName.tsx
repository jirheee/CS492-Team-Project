import {
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper
} from '@chakra-ui/react';
import InputWithFieldName from './InputWithFieldName';

interface NumberInputProps {
  id: string;
  label: string;
  defaultValue: number;
  min?: number;
}

const NumberInputWithFieldname = ({
  label,
  id,
  defaultValue,
  min
}: NumberInputProps) => {
  return (
    <InputWithFieldName label={label} id={id}>
      <NumberInput id={id} size="sm" defaultValue={defaultValue} min={min}>
        <NumberInputField />
        <NumberInputStepper>
          <NumberIncrementStepper />
          <NumberDecrementStepper />
        </NumberInputStepper>
      </NumberInput>
    </InputWithFieldName>
  );
};

export default NumberInputWithFieldname;
