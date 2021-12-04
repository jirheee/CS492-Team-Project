import { Layer } from '../../model/types';
import BaseBlock, { BlockProps } from './baseBlock';

const ActivationBlock = ({
  layerType,
  onClick,
  onClose,
  onModify
}: Omit<BlockProps<Layer>, 'popover'>) => {
  return (
    <BaseBlock<Layer>
      layerType={layerType}
      onClick={onClick}
      onClose={onClose}
      onModify={onModify}
    />
  );
};

export default ActivationBlock;
