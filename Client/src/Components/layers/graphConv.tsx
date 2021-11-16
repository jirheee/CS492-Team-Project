import { Input, Text } from '@chakra-ui/react';
import { useState } from 'react';
import BaseBlock from './baseBlock';
import { BiasInfo, LayerType } from '../../model/types';

const GraphConv = ({
  layerType,
  index
}: {
  layerType: LayerType;
  index: number;
}) => {
  const [channels, setChannels] = useState(1);
  const [bias, setBias] = useState(BiasInfo.True);
  return (
    <BaseBlock layerType={layerType} index={index}>
      <Text fontSize="12px">{`Channels: ${channels}`}</Text>
      <Text
        fontSize="12px"
        onClick={() => {
          setBias(b => (b === BiasInfo.True ? BiasInfo.False : BiasInfo.True));
        }}
        _hover={{ cursor: 'pointer', backgroundColor: 'gray.50' }}
      >{`Bias: ${bias}`}</Text>
    </BaseBlock>
  );
};

export default GraphConv;
