import { Input, Text } from '@chakra-ui/react';
import { useState } from 'react';
import BaseBlock from './baseBlock';
import { BiasInfo, LayerType } from '../../model/types';

const GraphConv = ({ layerType, isEditable }) => {
  const [channels, setChannels] = useState(1);
  const [bias, setBias] = useState(BiasInfo.True);
  return (
    <BaseBlock layerType={layerType} isEditable={isEditable}>
      <Text fontSize="12px">{`Channels: ${channels}`}</Text>
      <Text
        fontSize="12px"
        onClick={() => {
          isEditable &&
            setBias(b =>
              b === BiasInfo.True ? BiasInfo.False : BiasInfo.True
            );
        }}
        _hover={
          isEditable ? { cursor: 'pointer', backgroundColor: 'gray.50' } : {}
        }
      >{`Bias: ${bias}`}</Text>
    </BaseBlock>
  );
};

export default GraphConv;
