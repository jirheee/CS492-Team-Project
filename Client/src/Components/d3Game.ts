import { select, path } from 'd3';
import { Dispatch, SetStateAction } from 'react';

interface D3GameProps {
  boardColor: string;
  boardWidth: number;
  setBoard: Dispatch<SetStateAction<{ index: number; value: any }[]>>;
}

export enum CoordState {
  WHITE = 'white',
  BLACK = 'black',
  NONE = 'rgba(0,0,0,0)'
}

class D3Game {
  public boardWidth: number;
  public divEl: any;
  public svg: any;
  public boardColor: string;
  public setBoard: Dispatch<SetStateAction<{ index: number; value: any }[]>>;

  constructor(divEl: any, props: D3GameProps) {
    const { boardColor, boardWidth, setBoard } = props;
    this.boardWidth = boardWidth;
    this.divEl = divEl;
    this.setBoard = setBoard;

    this.svg = select(divEl)
      .append('svg')
      .style('background-color', boardColor)
      .attr('width', '100%')
      .attr('height', '100%')
      .style('padding', 30);

    this.boardColor = boardColor;
    this.initializeBoard();
  }

  public handleClick() {}

  public initializeBoard() {
    const svg = this.svg;

    const lineData: { start: [number, number]; end: [number, number] }[] =
      new Array(this.boardWidth * 2).fill(0).map((_, i) =>
        i < this.boardWidth
          ? {
              start: [60, 60 * i + 60],
              end: [60 * this.boardWidth, 60 * i + 60]
            }
          : {
              start: [60 * (i % this.boardWidth) + 60, 60],
              end: [60 * (i % this.boardWidth) + 60, 60 * this.boardWidth]
            }
      );

    function drawHorizontalLine(
      context: any,
      d: { start: [number, number]; end: [number, number] }
    ) {
      context.moveTo(d.start[0], d.start[1]);
      context.lineTo(d.end[0], d.end[1]);
      context.closePath();
      return context;
    }

    svg
      .selectAll('path')
      .data(lineData)
      .join(
        (enter: any) => enter.append('path'),
        (exit: any) => exit.remove()
      )
      .attr('d', (d: any) => drawHorizontalLine(path(), d))
      .attr('stroke', 'black')
      .attr('stroke-width', 1);
  }

  public renderGoStones(board: { index: number; value: CoordState }[]) {
    this.svg
      .selectAll('circle')
      .data(board)
      .join(
        enter => enter.append('circle'),
        exit => exit.remove()
      )
      .attr('cx', (_, i) => Math.floor(i % this.boardWidth) * 60 + 60)
      .attr('cy', (_, i) => Math.floor(i / this.boardWidth) * 60 + 60)
      .attr('r', 20)
      .attr('stroke', d => d)
      .attr('fill', d => d.value)
      .on('mouseover', function () {
        select(this).attr('fill', 'rgba(0,0,0,0.2)');
      })
      .on('mouseout', function (e, d) {
        select(this).attr('fill', d.value);
      })
      .on('mouseup', (e, d) => {
        console.log(this.setBoard);
        this.setBoard(board => {
          const { index, value } = d;
          const newBoard = [...board];
          newBoard[index].value =
            value === CoordState.NONE ? CoordState.BLACK : value;
          console.log(newBoard[index]);
          return newBoard;
        });
      });
  }
}

export default D3Game;
