import { useEffect, useState } from 'react';
import { Box, Flex, Text } from '@chakra-ui/react';
import Game from '../components/game';
import { useSocket } from '../lib/socket';

enum GameStatus {
  NOT_STARTED,
  PLAYING,
  END
}

interface PlayerProfileProps {
  name: string;
  isMyTurn: boolean;
  move: number;
  boardWidth: number;
  color: string;
}

const PlayerProfile = ({
  name,
  isMyTurn,
  move,
  boardWidth,
  color
}: PlayerProfileProps) => {
  return (
    <Flex
      borderColor={isMyTurn ? 'red.500' : 'gray.100'}
      w="400px"
      h="250px"
      p={3}
      m={5}
      borderWidth="thick"
      borderRadius="10px"
      flexDir="column"
      backgroundColor={color}
    >
      <Text fontWeight="bold">Player: {name}</Text>
      <Text>
        GoStone Color: {color === 'rgba(255,255,255)' ? 'White' : 'Black'}
      </Text>
      <Text>
        Move: {Math.floor(move / boardWidth) + 1} {(move % boardWidth) + 1}
      </Text>
    </Flex>
  );
};

const ShowWinner = ({ player, playerName }) => {
  console.log('Show Winner');
  return (
    <Flex
      w="100vw"
      h="100vh"
      position="fixed"
      justifyContent="center"
      alignItems="center"
      backgroundColor="rgba(0,0,0,0.2)"
      left={0}
      top={0}
    >
      <Text
        fontWeight="bold"
        fontSize="50px"
        textAlign="center"
        backgroundColor="rgba(255,255,255, 0.5)"
        p={5}
      >
        🎉 Player {player} {playerName} Won! 🎉
      </Text>
    </Flex>
  );
};

const BattlePage = () => {
  console.log('BattlePage');
  const { socket, connected } = useSocket();
  const [player1Name, setPlayer1Name] = useState('');
  const [player2Name, setPlayer2Name] = useState('');
  const [nInRow, setNInRow] = useState(0);
  const [boardWidth, setBoardWidth] = useState(0);
  const [gameStatus, setGameStatus] = useState(GameStatus.NOT_STARTED);

  const [previousPlayer1Move, setPreviousePlayer1Move] = useState({ move: -1 });
  const [previousPlayer2Move, setPreviousePlayer2Move] = useState({ move: -1 });

  const [currentPlayer, setCurrentPlayer] = useState(0);

  const [winner, setWinner] = useState(0);

  useEffect(() => {
    if (socket && connected) {
      socket.on(
        'BattleStart',
        ({ player1, player2, board_width, n_in_row }) => {
          console.log(player1, player2);
          setPlayer1Name(player1);
          setPlayer2Name(player2);
          setBoardWidth(board_width);
          setNInRow(n_in_row);
          setGameStatus(GameStatus.PLAYING);
        }
      );
      socket.on('Winner', ({ player }) => {
        console.log(player);
        setGameStatus(GameStatus.END);
        setWinner(player);
      });
      socket.on('Move', ({ player, move }) => {
        if (player === 1) {
          setPreviousePlayer1Move({ move });
          setCurrentPlayer(1);
        } else {
          setPreviousePlayer2Move({ move });
          setCurrentPlayer(2);
        }
      });
    }
  }, [connected, socket]);

  return (
    <>
      <Flex
        w="100vw"
        h="100vh"
        alignContent="center"
        justifyContent="conter"
        flexDir="column"
      >
        <Flex p={10} justifyContent="space-evenly" backgroundColor="gray.100">
          <Text fontWeight="bold">Board Width: {boardWidth}</Text>
          <Text fontWeight="bold">N in Row: {nInRow}</Text>
        </Flex>
        <Flex alignContent="center" justifyContent="center" p={5}>
          {gameStatus !== GameStatus.NOT_STARTED && (
            <Box
              w={`${(boardWidth * 600) / 8 + 30}px`}
              h={`${(boardWidth * 600) / 8 + 30}px`}
              borderColor="gray.100"
              borderWidth="medium"
              borderRadius="30px"
              overflow="hidden"
              m={3}
            >
              <Game boardWidth={boardWidth} agentUuid="" />
            </Box>
          )}
          <Flex flexDir="column">
            <PlayerProfile
              name={player2Name}
              isMyTurn={currentPlayer === 2}
              move={previousPlayer2Move.move}
              boardWidth={boardWidth}
              color="rgba(0,0,0,0.1)"
            />
            <PlayerProfile
              name={player1Name}
              isMyTurn={currentPlayer === 1}
              move={previousPlayer1Move.move}
              boardWidth={boardWidth}
              color="rgba(255,255,255)"
            />
          </Flex>
        </Flex>
      </Flex>
      {gameStatus === GameStatus.END && (
        <ShowWinner
          player={winner}
          playerName={winner === 1 ? player1Name : player2Name}
        />
      )}
    </>
  );
};

export default BattlePage;
