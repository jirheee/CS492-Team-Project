import { Link } from "react-router-dom";
import HoverBoxButton from "../Components/HoverBoxButton";

const HomePage = () => {
  return (
    <div className="HoverBoxBtnConatiner">
      <Link to="/model">
        <HoverBoxButton
          text="Model"
          imgSrc={null}
          hoverText="Build & Train you own Agent"
        />
      </Link>
      <Link to="/battle">
        <HoverBoxButton
          text="Battle"
          imgSrc={null}
          hoverText="Battle your agents with other Agents"
        />
      </Link>
    </div>
  );
};

export default HomePage;
