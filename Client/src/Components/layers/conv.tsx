import { Input, Text } from '@chakra-ui/react';
import { useState } from 'react';
import BaseBlock from './baseBlock';
import { BiasInfo, ConvLayer, LayerType } from '../../model/types';

const Conv = ({ index }) => {
  const [channels, setChannels] = useState(1);
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(1);
  const [kernel_size, setKernelSize] = useState(1);
  const [bias, setBias] = useState(BiasInfo.True);
  return (
    <BaseBlock layerType={LayerType.Conv} index={index}>
      <Text fontSize="12px">{`Channels: ${channels}`}</Text>
      <Text fontSize="12px">{`Kernel Size: ${kernel_size}`}</Text>
      <Text fontSize="12px">{`Stride: ${stride}`}</Text>
      <Text fontSize="12px">{`Padding: ${padding}`}</Text>
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

export default Conv;
