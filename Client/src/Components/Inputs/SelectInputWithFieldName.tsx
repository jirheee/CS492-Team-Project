import { Select } from '@chakra-ui/react';
import InputWithFieldName from './InputWithFieldName';

const SelectInputWithFieldName = ({ children, label, id }) => {
  return (
    <InputWithFieldName label={label} id={id}>
      <Select id={id}>{children}</Select>
    </InputWithFieldName>
  );
};

export default SelectInputWithFieldName;
