import { Server as SocketIOServer } from 'socket.io';
import { Not } from 'typeorm';
import Agent from '../entity/agent';
import TrainManager from '../manager/trainManager';
import PythonSpawner from '../ml/pythonSpawner';
import { getAgentModel, getAgentTrainHistory } from '../routes/agent/logic';
import { ProcessType, RandomBattleRequest } from '../types';
import { TrainStatus } from '../types/nn';

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
      'BattleStart',
      async ({ agentUuid, opponent }: RandomBattleRequest) => {
        const model = await getAgentModel(agentUuid);

        /* eslint-disable camelcase */
        const {
          board: { board_width, board_height, n_in_row },
          name
        } = model;

        const randomAgent = await (async () => {
          const agents = await Agent.find({
            where: { trainStatus: Not(TrainStatus.NOT_TRAINED) }
          });
          console.log(agents);
          return agents[Math.floor(Math.random() * agents.length) - 1];
        })();

        console.log(randomAgent.name);

        const opponentUuid =
          opponent === 'Random' ? randomAgent.uuid : opponent;

        const gameOptions = [
          '--player1',
          agentUuid,
          '--player2',
          opponentUuid,
          '--board_width',
          String(board_width),
          '--board_height',
          String(board_height),
          '--n_in_row',
          String(n_in_row)
        ];

        const onData = async (data: string) => {
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

              const winner = await Agent.findOne(
                player === 1 ? agentUuid : opponentUuid
              );

              winner!.win += 1;

              await winner?.save();

              const loser = await Agent.findOne(
                player === 1 ? opponentUuid : agentUuid
              );
              loser!.lose += 1;
              await loser?.save();
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

        process.run().catch(e => console.log(e));

        socket.emit('BattleStart', {
          player1: name,
          player2: (await Agent.findOne(opponentUuid))?.name,
          board_width,
          n_in_row
        });
      }
    );

    socket.on('MonitorTrainHistory', async ({ agentUuid }) => {
      const agent = await Agent.findOne(agentUuid);
      console.log(agent);
      if (agent?.trainStatus === TrainStatus.TRAIN_FINISHED) {
        const trainOutput = await getAgentTrainHistory(agentUuid);
        const { train_progression, win_rates } = trainOutput;
        console.log(
          win_rates.map(log => {
            return { epoch: log[0], win_rate: log[1] };
          })
        );
        socket.emit('History', {
          trainHistory: train_progression.map(log => {
            return {
              epoch: log[0],
              elapsed_time: log[1],
              loss: log[2],
              entropy: log[3],
              kl: log[4]
            };
          }),
          winRateHistory: win_rates.map(log => {
            return { epoch: log[0], win_rate: log[1] };
          })
        });
        return;
      }
      const trainProcess =
        TrainManager.getInstance().trainingProcess.get(agentUuid);

      if (trainProcess) {
        console.log(trainProcess.getTrainingHistory());
        socket.emit('History', {
          trainHistory: trainProcess.getTrainingHistory(),
          winRateHistory: trainProcess.getWinRateHistory()
        });
        trainProcess.events.onData = () => {
          socket.emit('History', {
            trainHistory: trainProcess.getTrainingHistory(),
            winRateHistory: trainProcess.getWinRateHistory()
          });
        };
      } else {
        console.error('No such process');
      }
    });
  });
};
