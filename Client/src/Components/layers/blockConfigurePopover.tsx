import React from 'react';
import {
  Button,
  Popover,
  PopoverTrigger,
  PopoverArrow,
  PopoverContent,
  IconButton,
  PopoverHeader,
  PopoverBody,
  Grid
} from '@chakra-ui/react';
import { EditIcon } from '@chakra-ui/icons';
import { BiasInfo, ConvLayer, GraphConvLayer } from '../../model/types';
import {
  getMinimunvalueForField,
  pascalCaseToSnakeCase,
  snakeCaseToPascalCase
} from '../../lib/util';
import NumberInputWithFieldname from '../Inputs/NumberInputWithFieldName';
import SelectInputWithFieldName from '../Inputs/SelectInputWithFieldName';

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
