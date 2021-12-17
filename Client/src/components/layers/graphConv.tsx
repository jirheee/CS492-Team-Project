import { Text } from '@chakra-ui/react';
import BaseBlock, { BlockProps } from './baseBlock';
import { GraphConvLayer } from '../../model/types';
import BlockConfigurePopover from './blockConfigurePopover';

const GraphConv = ({
  layerType,
  layerProps,
  onClick,
  onClose,
  onModify
}: BlockProps<GraphConvLayer> & { layerProps: GraphConvLayer }) => {
  const { channels, bias } = layerProps;
  return (
    <BaseBlock<GraphConvLayer>
      layerType={layerType}
      popover={
        <BlockConfigurePopover layerProps={layerProps} onModify={onModify} />
      }
      onClick={onClick}
      onClose={onClose}
      onModify={onModify}
    >
      <Text fontSize="12px">{`Channels: ${channels}`}</Text>
      <Text fontSize="12px">{`Bias: ${bias}`}</Text>
    </BaseBlock>
  );
};

export default GraphConv;
