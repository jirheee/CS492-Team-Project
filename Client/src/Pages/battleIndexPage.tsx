import BaseIndexPage from '../components/baseIndexPage';
import HoverBtn from '../components/hoverBtn';

const BattleIndexPage = () => {
  return (
    <BaseIndexPage>
      <HoverBtn baseText="Random" hoverText="Battle with Random Opponent" />
      <HoverBtn baseText="Choose" hoverText="Choose an Agent to Battle" />
      <HoverBtn baseText="Yourself" hoverText="Battle with your Agent" />
    </BaseIndexPage>
  );
};

export default BattleIndexPage;
