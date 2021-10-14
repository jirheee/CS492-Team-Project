import { Link, Route } from "react-router-dom";
import HoverBoxButton from "../Components/HoverBoxButton";
// import NeuralNet from "../Components/Neuralnet/Neuralnet";

const TrainPage = () => {
  return <h1>Train Model</h1>;
};

const CreatePage = () => {
  return <h1>Create Model</h1>;
};

const CreateTrainSelection = ({ match }) => {
  return (
    <div className="HoverBoxBtnConatiner">
      <Link to={`${match.path}/create`}>
        <HoverBoxButton
          text="Create"
          imgSrc={null}
          hoverText="Build an Agent"
        />
      </Link>
      <Link to={`${match.path}/train`}>
        <HoverBoxButton
          text="Train"
          imgSrc={null}
          hoverText="Train your Agent"
        />
      </Link>
    </div>
  );
};

const ModelPage = ({ match }) => {
  return (
    <>
      <Route exact path={match.path} component={CreateTrainSelection} />
      <Route path={`${match.path}/create`} component={CreatePage} />
      <Route path={`${match.path}/train`} component={TrainPage} />
    </>
  );
};

export default ModelPage;
