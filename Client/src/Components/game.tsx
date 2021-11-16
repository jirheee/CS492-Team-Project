import React, { useState, useRef, useEffect, useLayoutEffect } from 'react';
import { select, path } from 'd3';
import D3Game, { CoordState } from './d3Game';
import { Box } from '@chakra-ui/layout';

let game: D3Game;

const Game = ({ boardWidth }: { boardWidth: number }) => {
  const boardColor = 'rgb(218, 179, 79)';

  const boardRef = useRef<HTMLDivElement | null>(null);

  const [board, setBoard] = useState(
    new Array(boardWidth * boardWidth).fill(CoordState.NONE).map((v, i) => {
      return { index: i, value: v };
    })
  );

  useEffect(() => {
    if (board && board.length) {
      const GameProps = {
        boardColor,
        boardWidth,
        setBoard
      };

      game = new D3Game(boardRef.current, GameProps);
      game.renderGoStones(board);
    }
  }, [board, boardWidth]);

  return <Box ref={boardRef} w="full" h="100vh" />;
};

export default Game;
