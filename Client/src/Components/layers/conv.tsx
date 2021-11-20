import { Input, Text } from '@chakra-ui/react';
import { useState } from 'react';
import BaseBlock from './baseBlock';
import { BiasInfo, ConvLayer, LayerType } from '../../model/types';

<<<<<<< HEAD
const Conv = ({ isEditable }) => {
=======
const Conv = ({ index }) => {
>>>>>>> 8d2740530bc68f98590e309604a10605395c3100
  const [channels, setChannels] = useState(1);
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(1);
  const [kernel_size, setKernelSize] = useState(1);
  const [bias, setBias] = useState(BiasInfo.True);
  return (
    <BaseBlock layerType={LayerType.Conv} isEditable={isEditable}>
      <Text fontSize="12px">{`Channels: ${channels}`}</Text>
      <Text fontSize="12px">{`Kernel Size: ${kernel_size}`}</Text>
      <Text fontSize="12px">{`Stride: ${stride}`}</Text>
      <Text fontSize="12px">{`Padding: ${padding}`}</Text>
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

export default Conv;
