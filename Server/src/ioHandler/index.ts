import { Server as SocketIOServer } from 'socket.io';
import PythonSpawner from '../ml/pythonSpawner';
import { getAgentModel } from '../routes/agent/logic';
import { ProcessType, RandomBattleRequest } from '../types';

export default (io: SocketIOServer) => {
  io.on('connection', socket => {
    console.log('connection');

    socket.on('disconnect', () => {
      console.log('disconnected');
      socket.disconnect();
      io.emit('user', 'disconnected');
    });

    socket.on('handshake', msg => {
      console.log(msg);
      socket.emit('handshake', msg + msg);
    });

    socket.on(
      'RandomBattleStart',
      async ({ agentUuid }: RandomBattleRequest) => {
        const model = await getAgentModel(agentUuid);

        /* eslint-disable camelcase */
        const {
          board: { board_width, board_height, n_in_row },
          name
        } = model;

        const gameOptions = [
          '--player1',
          agentUuid,
          '--player2',
          agentUuid,
          '--board_width',
          String(board_width),
          '--board_height',
          String(board_height),
          '--n_in_row',
          String(n_in_row)
        ];

        const onData = (data: string) => {
          console.log(data);
          // socket.emit('Move', move);
          try {
            const parsedObject = JSON.parse(data);
            if (parsedObject.action === 'move') {
              const { player, move } = parsedObject;
              socket.emit('Move', { player, move });
            }
            if (parsedObject.action === 'winner') {
              const { player } = parsedObject;
              socket.emit('Winner', { player });
            }
          } catch (e) {
            console.log('err');
          }
        };

        const process = new PythonSpawner(
          './src/ml/AlphaZero_Gomoku',
          'game.py',
          gameOptions,
          { onData },
          ProcessType.Battle
        );

        // let i = 0;
        // const sendMove = () => {
        //   console.log(i);
        //   socket.emit('Move', i);
        //   i += 1;
        //   setTimeout(sendMove, 2000);
        // };

        // setTimeout(sendMove, 2000);

        process.run().catch(e => console.log(e));

        socket.emit('BattleStart', {
          player1: name,
          player2: name,
          board_width,
          n_in_row
        });
      }
    );
  });
};
