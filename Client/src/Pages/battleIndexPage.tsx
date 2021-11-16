import { Link } from 'react-router-dom';
import BaseIndexPage from '../components/baseIndexPage';
import HoverBtn from '../components/hoverBtn';

const BattleIndexPage = () => {
  return (
    <BaseIndexPage>
      <Link to="/battle/game">
        <HoverBtn baseText="Random" hoverText="Battle with Random Opponent" />
      </Link>
      <HoverBtn baseText="Choose" hoverText="Choose an Agent to Battle" />
      <HoverBtn baseText="Yourself" hoverText="Battle with your Agent" />
    </BaseIndexPage>
  );
};

export default BattleIndexPage;
