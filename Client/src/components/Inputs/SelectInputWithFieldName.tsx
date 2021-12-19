import { Select } from '@chakra-ui/react';
import InputWithFieldName from './InputWithFieldName';

const SelectInputWithFieldName = ({
  children,
  label,
  id,
  onChange
}: {
  children;
  label;
  id;
  onChange?;
}) => {
  return (
    <InputWithFieldName label={label} id={id}>
      <Select id={id} onChange={onChange}>
        {children}
      </Select>
    </InputWithFieldName>
  );
};

export default SelectInputWithFieldName;
