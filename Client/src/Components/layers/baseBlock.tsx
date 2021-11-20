import React from 'react';
import { Flex, Text, CloseButton } from '@chakra-ui/react';
import { LayerType } from '../../model/types';
import { getBlockColor } from '../../lib/util';
import { ReactElement } from 'react-router/node_modules/@types/react';

export interface BlockProps<T> {
  layerType: LayerType;
  popover?: ReactElement;
  onClick?: () => void;
  onClose?: () => void;
  onModify?: (newLayer: T) => () => void;
}

const BaseBlock = <T extends unknown>({
  children,
  layerType,
  popover,
  onClick,
  onClose
}: React.PropsWithChildren<BlockProps<T>>) => {
  return (
    <Flex
      w={100}
      h={160}
      backgroundColor={getBlockColor(layerType)}
      borderWidth="1px"
      position="relative"
      borderRadius="8px"
      justifyContent="center"
      flexDir="column"
      onClick={onClick}
      _hover={onClick ? { cursor: 'pointer' } : {}}
    >
      {!onClick && (
        <Flex
          w="full"
          justifyContent="space-around"
          position="absolute"
          right={0}
          top={0}
        >
          {popover}
          <CloseButton onClick={onClose} marginLeft="auto" />
        </Flex>
      )}
      <Flex flexDir="column" textAlign="center" justifyContent="center">
        <Text fontWeight="bold">{layerType}</Text>
        {children}
      </Flex>
    </Flex>
  );
};

export default BaseBlock;
