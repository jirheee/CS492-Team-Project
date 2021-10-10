import { IconButton } from "@material-ui/core";
import { AddCircle, RemoveCircle } from "@material-ui/icons";
import { useState } from "react";
import Layer from "../Layer/Layer";
import "./Network.css";

const Network = () => {
  const [numLayers, setNumLayers] = useState(2);

  const IncreaseLayer = () => {
    setNumLayers(numLayers + 1);
  };

  const DecreaseLayer = () => {
    if (numLayers > 2) {
      setNumLayers(numLayers - 1);
    }
  };

  const Layers = Array.from(Array(numLayers)).map((_, i) => <Layer key={i} />);

  return (
    <>
      <div>
        <IconButton aria-label="Add" onClick={IncreaseLayer}>
          <AddCircle />
        </IconButton>
        <IconButton aria-label="Delete" onClick={DecreaseLayer}>
          <RemoveCircle />
        </IconButton>
      </div>
      <div className="network">{Layers}</div>
    </>
  );
};

export default Network;
