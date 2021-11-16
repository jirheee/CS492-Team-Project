import React, { useContext, useRef, useState } from 'react';
import { useEffect } from 'react-router/node_modules/@types/react';
import Model from '../../model/model';
import { LayerType } from '../../model/types';
import ModelContext from './ModelContext';

const ModelProvider = ({ children }: React.PropsWithChildren<{}>) => {
  const modelRef = useRef(new Model());

  const addLayer = (layerType: LayerType) => {
    if (modelRef.current) {
      console.log(`Create Layer: ${layerType}`);
      modelRef.current.addLayer(layerType);
    }
  };

  const removeLayer = (index: number) => {
    if (modelRef.current) {
      console.log(`Remove ${index}th Layer`);
      modelRef.current.removeLayer(index);
    }
  };

  const modifyLayer = (index, props) => {
    if (modelRef.current) {
      console.log(
        `Modify ${index}th Layer with props: ${JSON.stringify(props)}`
      );
      modelRef.current.modifyLayer(index, props);
    }
  };

  return (
    <ModelContext.Provider
      value={{ model: modelRef.current, addLayer, modifyLayer, removeLayer }}
    >
      {children}
    </ModelContext.Provider>
  );
};

export default ModelProvider;
