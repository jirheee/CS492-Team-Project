import { Link, Route } from "react-router-dom";
import HoverBoxButton from "../Components/HoverBoxButton";

const InGame = () => {
  return <h1>InGame</h1>;
};

const MatchTypeSelection = ({ match }) => {
  return (
    <div className="HoverBoxBtnConatiner">
      <Link to={`${match.path}/ingame`}>
        <HoverBoxButton
          text="Random Match"
          imgSrc={null}
          hoverText="Random match with other agents"
        />
        <HoverBoxButton
          text="Choose Opponent"
          imgSrc={null}
          hoverText="Choose an agent you want to battle"
        />
        <HoverBoxButton
          text="Battle your agent!"
          imgSrc={null}
          hoverText="Battle your Agent"
        />
      </Link>
    </div>
  );
};

const BattlePage = ({ match }) => {
  return (
    <>
      <Route exact path={match.path} component={MatchTypeSelection} />
      <Route path={`${match.path}/ingame`} component={InGame} />
    </>
  );
};

export default BattlePage;
