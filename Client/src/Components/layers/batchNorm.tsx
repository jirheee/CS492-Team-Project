import { LayerType } from '../../model/types';
import BaseBlock from './baseBlock';

const BatchNorm = ({ isEditable }) => {
  return <BaseBlock layerType={LayerType.BatchNorm} isEditable={isEditable} />;
};

export default BatchNorm;
