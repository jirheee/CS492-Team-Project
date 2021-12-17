import { Slider } from '@chakra-ui/react';
import InputWithFieldName from './InputWithFieldName';

interface SliderInputProps {
  label: string;
  id: string;
  min: number;
  max: number;
}

const SliderWithFieldName = ({ label, id, min, max }: SliderInputProps) => {
  return (
    <InputWithFieldName label={label} id={id}>
      <Slider min={min} max={max} />
    </InputWithFieldName>
  );
};

export default SliderWithFieldName;
