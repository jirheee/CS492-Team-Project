import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import BaseIndexPage from '../components/baseIndexPage';
import HoverBtn from '../components/hoverBtn';
import { useSocket } from '../lib/socket';

const BattleIndexPage = () => {
  const { socket } = useSocket();

  const emitRandomBattleStart = () => {
    console.log('Emit');
    socket?.emit('RandomBattleStart', {
      agentUuid: '80895d19-1a04-4821-b57c-2264ac7d3194'
    });
  };
  return (
    <BaseIndexPage>
      <Link to="/battle/game" onClick={emitRandomBattleStart}>
        <HoverBtn baseText="Random" hoverText="Battle with Random Opponent" />
      </Link>
      <HoverBtn baseText="Choose" hoverText="Choose an Agent to Battle" />
      <HoverBtn baseText="Yourself" hoverText="Battle with your Agent" />
    </BaseIndexPage>
  );
};

export default BattleIndexPage;
