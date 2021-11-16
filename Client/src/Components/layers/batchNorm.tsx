import { LayerType } from '../../model/types';
import BaseBlock from './baseBlock';

const BatchNorm = ({ index }) => {
  return <BaseBlock layerType={LayerType.BatchNorm} index={index} />;
};

export default BatchNorm;
