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
  max?: number;
  step?: number;
}

const NumberInputWithFieldname = ({
  label,
  id,
  defaultValue,
  min,
  max,
  step
}: NumberInputProps) => {
  return (
    <InputWithFieldName label={label} id={id}>
      <NumberInput
        id={id}
        size="sm"
        defaultValue={defaultValue}
        min={min}
        step={step}
        max={max}
      >
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
