import { select, path } from 'd3';
import { Socket } from 'socket.io-client';

interface Coord {
  index: number;
  value: CoordState;
}

interface D3GameProps {
  boardColor: string;
  boardWidth: number;
}

export enum CoordState {
  WHITE = 'white',
  BLACK = 'black',
  NONE = 'rgba(0,0,0,0)'
}

type Move = {
  player: 1 | 2;
  move: number;
};

class D3Game {
  public boardWidth: number;
  public divEl: any;
  public svg: any;
  public boardColor: string;
  public socket: Socket;
  public board: Coord[];
  public myAgentUuid: string;

  constructor(
    divEl: any,
    props: D3GameProps,
    socket: Socket,
    agentUuid: string
  ) {
    const { boardColor, boardWidth } = props;
    this.boardWidth = boardWidth;
    this.divEl = divEl;
    this.socket = socket;
    this.board = new Array(boardWidth * boardWidth)
      .fill(CoordState.NONE)
      .map((v, i) => {
        return { index: i, value: v };
      });

    console.log('D3Game');

    this.svg = select(divEl)
      .append('svg')
      .style('background-color', boardColor)
      .attr('width', '100%')
      .attr('height', '100%')
      .style('padding', 30);

    this.boardColor = boardColor;
    this.initializeBoard();

    this.myAgentUuid = agentUuid;

    socket.emit('StartRandomBattle');
  }

  public registerSocketEvents() {
    const { socket } = this;

    socket.on('move', () => {});
  }

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

  public renderGoStones() {
    this.svg.selectAll('svg > circle').remove();

    this.svg
      .selectAll('circle')
      .data(this.board)
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
        const { index, value } = d;
        console.log(index);
        this.board[index] = { index, value: CoordState.BLACK };
      });
  }
}

export default D3Game;
