import { Input } from '@chakra-ui/react';
import InputWithFieldName from './InputWithFieldName';

interface TextInputProps {
  label: string;
  id: string;
  placeholder: string;
  layout?: string;
  onChange?: (e) => void;
}

const TextInputWithFieldName = ({
  label,
  id,
  placeholder,
  layout = '1fr 1fr',
  onChange = undefined
}: TextInputProps) => {
  return (
    <InputWithFieldName label={label} id={id} layout={layout}>
      <Input placeholder={placeholder} onChange={onChange} />
    </InputWithFieldName>
  );
};

export default TextInputWithFieldName;
