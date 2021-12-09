import { Layer, LayerType } from '../../model/types';
import BaseBlock, { BlockProps } from './baseBlock';

const BatchNorm = ({
  onClick,
  onClose,
  onModify
}: Omit<BlockProps<Layer>, 'layerType'>) => {
  return (
    <BaseBlock<Layer>
      layerType={LayerType.BatchNorm}
      onClick={onClick}
      onClose={onClose}
      onModify={onModify}
    />
  );
};

export default BatchNorm;
