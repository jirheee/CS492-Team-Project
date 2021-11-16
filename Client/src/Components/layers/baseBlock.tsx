import React, { useState } from 'react';
import { Flex, Text, CloseButton } from '@chakra-ui/react';
import { LayerType } from '../../model/types';
import { getBlockColor } from '../../lib/util';

interface BlockProps {
  children?: React.ReactElement[];
  layerType: LayerType;
  index: number;
}

const BaseBlock = ({ children, layerType, index }: BlockProps) => {
  const [hover, setHover] = useState(false);
  return (
    <Flex
      w={100}
      h={150}
      backgroundColor={getBlockColor(layerType)}
      borderWidth="1px"
      alignContent="center"
      justifyContent="center"
      flexDir="column"
      textAlign="center"
      position="relative"
      onMouseEnter={() => {
        setHover(true);
      }}
      onMouseLeave={() => {
        setHover(false);
      }}
    >
      {hover && (
        <CloseButton
          onClick={() => {
            console.log(index);
          }}
          position="absolute"
          right={0}
          top={0}
        />
      )}
      <Text fontWeight="bold">{layerType}</Text>
      {children}
    </Flex>
  );
};

export default BaseBlock;
