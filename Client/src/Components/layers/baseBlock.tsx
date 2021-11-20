import React, { useState } from 'react';
import { Flex, Text, CloseButton } from '@chakra-ui/react';
import { LayerType } from '../../model/types';
import { getBlockColor } from '../../lib/util';

interface BlockProps {
  children?: React.ReactElement[];
  layerType: LayerType;
  isEditable: boolean;
}

const BaseBlock = ({ children, layerType, isEditable }: BlockProps) => {
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
        isEditable && setHover(true);
      }}
      onMouseLeave={() => {
        isEditable && setHover(false);
      }}
    >
      {hover && isEditable && (
        <CloseButton position="absolute" right={0} top={0} />
      )}
      <Text fontWeight="bold">{layerType}</Text>
      {children}
    </Flex>
  );
};

export default BaseBlock;
