import { FormControl, FormLabel } from '@chakra-ui/react';

interface InputProps {
  label: string;
  id: string;
  layout?: string;
}

const InputWithFieldName = ({
  label,
  id,
  children,
  layout = '1fr 1fr'
}: React.PropsWithChildren<InputProps>) => {
  return (
    <FormControl display="grid" gridTemplateColumns={layout} p={1}>
      <FormLabel htmlFor={id} w="full" h="full" m={0} p={1}>
        {label}
      </FormLabel>
      {children}
    </FormControl>
  );
};

export default InputWithFieldName;
