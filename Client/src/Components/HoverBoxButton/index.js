import { useState } from "react";
import "./HoverBoxButton.css";

const HoverBoxButton = props => {
  const [showHover, setShowHover] = useState(false);

  return (
    <button
      className="HoverBoxButton"
      onMouseEnter={() => setShowHover(true)}
      onMouseLeave={() => setShowHover(false)}
    >
      {showHover ? (
        <div>{props?.hoverText ?? "default hover text"}</div>
      ) : (
        <div>
          <img src={props?.img} alt=""></img>
          {props?.text ?? "default text"}
        </div>
      )}
    </button>
  );
};

export default HoverBoxButton;
