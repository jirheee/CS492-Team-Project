import { Text } from '@chakra-ui/react';
import BaseBlock, { BlockProps } from './baseBlock';
import { ConvLayer, LayerType } from '../../model/types';
import BlockConfigurePopover from './blockConfigurePopover';

const Conv = ({
  layerProps,
  onClick,
  onClose,
  onModify
}: Omit<BlockProps<ConvLayer>, 'layerType'> & { layerProps: ConvLayer }) => {
  const { stride, channels, kernel_size, padding, bias } = layerProps;
  return (
    <BaseBlock<ConvLayer>
      layerType={LayerType.Conv}
      popover={
        <BlockConfigurePopover layerProps={layerProps} onModify={onModify} />
      }
      onClick={onClick}
      onClose={onClose}
    >
      <Text fontSize="12px">{`Channels: ${channels}`}</Text>
      <Text fontSize="12px">{`Kernel Size: ${kernel_size}`}</Text>
      <Text fontSize="12px">{`Stride: ${stride}`}</Text>
      <Text fontSize="12px">{`Padding: ${padding}`}</Text>
      <Text fontSize="12px">{`Bias: ${bias}`}</Text>
    </BaseBlock>
  );
};

export default Conv;
