import Game from '../components/game';
import { SocketProvider } from '../lib/socket';

const BattlePage = () => {
  console.log('BattlePage');

  return (
    <SocketProvider url="localhost:5000">
      <Game boardWidth={5} agentUuid="" />
    </SocketProvider>
  );
};

export default BattlePage;
