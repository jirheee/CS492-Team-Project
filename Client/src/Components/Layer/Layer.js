import { IconButton } from "@material-ui/core";
import { AddCircle, RemoveCircle } from "@material-ui/icons";
import { useState } from "react";
import Node from "../Node/Node"

const Layer = () => {
    const [numNodes, setNumNodes] = useState(2);

    const Nodes = Array.from(Array(numNodes)).map((i) => <Node key={i}/>);

    const IncreaseNodes = () => {
        setNumNodes(numNodes + 1)
    }

    const DecreaseNodes = () => {
        setNumNodes(numNodes - 1)
    }

    return <div className="layer">
        <IconButton aria-label="Add" onClick={IncreaseNodes} color="primary">
            <AddCircle/>
        </IconButton>
        <IconButton aria-label="Delete" onClick={DecreaseNodes} color="secondary">
            <RemoveCircle/>
        </IconButton>
        {Nodes}
    </div>
}

export default Layer;