import React from 'react';
import {
  FormControl,
  FormLabel,
  Button,
  Popover,
  PopoverTrigger,
  PopoverArrow,
  PopoverContent,
  IconButton,
  PopoverHeader,
  PopoverBody,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Select,
  Grid
} from '@chakra-ui/react';
import { EditIcon } from '@chakra-ui/icons';
import { BiasInfo, ConvLayer, GraphConvLayer } from '../../model/types';
import {
  getMinimunvalueForField,
  pascalCaseToSnakeCase,
  snakeCaseToPascalCase
} from '../../lib/util';

interface NumberInputProps {
  id: string;
  label: string;
  defaultValue: number;
  min?: number;
}

interface InputProps {
  label: string;
  id: string;
}

const InputWithFieldName = ({
  label,
  id,
  children
}: React.PropsWithChildren<InputProps>) => {
  return (
    <FormControl display="grid" gridTemplateColumns="1fr 1fr" p={1}>
      <FormLabel htmlFor={id} w="full" h="full" m={0} p={1}>
        {label}
      </FormLabel>
      {children}
    </FormControl>
  );
};

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

const SelectInputWithFieldName = ({ children, label, id }) => {
  return (
    <InputWithFieldName label={label} id={id}>
      <Select id={id}>{children}</Select>
    </InputWithFieldName>
  );
};

interface BlockConfigurePopoverProps<T> {
  layerProps: T;
  onModify?: (layerProps: T) => () => void;
}

const BlockConfigurePopover = <T extends ConvLayer | GraphConvLayer>({
  layerProps,
  onModify
}: BlockConfigurePopoverProps<T>) => {
  const inputs = Object.keys(layerProps)
    .slice(1)
    .map(key => {
      const label = snakeCaseToPascalCase(key, ' ');
      return key !== 'bias' ? (
        <NumberInputWithFieldname
          label={label}
          id={label}
          defaultValue={layerProps[key]}
          min={getMinimunvalueForField(key)}
          key={label}
        />
      ) : (
        <SelectInputWithFieldName label={label} id={label} key={label}>
          <option value={BiasInfo.True}>True</option>
          <option value={BiasInfo.False}>False</option>
        </SelectInputWithFieldName>
      );
    });
  const handleSubmit = e => {
    e.preventDefault();
    const newLayer = Object.keys(layerProps)
      .slice(1)
      .map((_, i) => e.target[i])
      .reduce(
        (obj, t) => {
          return { ...obj, [pascalCaseToSnakeCase(t.id, ' ', '_')]: t.value };
        },
        { layer_name: layerProps.layer_name }
      );

    onModify && onModify(newLayer)();
  };
  return (
    <Popover placement="left">
      {({ isOpen, onClose }) => (
        <>
          <PopoverTrigger>
            <IconButton
              aria-label="Edit"
              icon={<EditIcon />}
              size="sm"
              backgroundColor="rgba(0,0,0,0)"
              _hover={{ backgroundColor: 'rgba(1,1,1,0.1)' }}
            />
          </PopoverTrigger>
          <PopoverContent>
            <PopoverArrow />
            <PopoverHeader>Edit Hyperparameters</PopoverHeader>
            <PopoverBody
              as="form"
              onSubmit={e => {
                handleSubmit(e);
                onClose();
              }}
            >
              {inputs}
              <Grid templateColumns="1fr 1fr" gap={1}>
                <Button colorScheme="teal" type="submit">
                  Apply
                </Button>
                <Button colorScheme="gray" onClick={onClose}>
                  Cancel
                </Button>
              </Grid>
            </PopoverBody>
          </PopoverContent>
        </>
      )}
    </Popover>
  );
};

export default BlockConfigurePopover;
