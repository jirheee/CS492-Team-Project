import { createContext } from 'react';
import Model from '../../model/model';
import { LayerType } from '../../model/types';

interface ModelContextInterface {
  model: Model | null;
  addLayer: (layerType: LayerType) => void;
  removeLayer: (index) => void;
  modifyLayer: (index, props) => void;
}

const ModelContext = createContext<ModelContextInterface>({
  model: null,
  addLayer: _ => {},
  removeLayer: index => {},
  modifyLayer: (index, props) => {}
});

export default ModelContext;
