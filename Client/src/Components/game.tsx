import React, { useRef, useEffect } from 'react';
import D3Game from './d3Game';
import { Box, Button } from '@chakra-ui/react';
import { useSocket } from '../lib/socket';

let game: D3Game;

const Game = ({
  boardWidth,
  agentUuid
}: {
  boardWidth: number;
  agentUuid: string;
}) => {
  const boardColor = 'rgb(218, 179, 79)';

  console.log('Game');

  const boardRef = useRef<HTMLDivElement | null>(null);

  const { socket, connected } = useSocket();

  useEffect(() => {
    if (connected && game === undefined && socket) {
      const GameProps = {
        boardColor,
        boardWidth
      };

      game = new D3Game(boardRef.current, GameProps, socket, agentUuid);
      game.renderGoStones();
    }
  }, [connected, boardWidth, socket, agentUuid]);

  return (
    <>
      <Box ref={boardRef} w="full" h="100vh" />
      <Button onClick={() => game.renderGoStones()}>Render</Button>
    </>
  );
};

export default Game;
