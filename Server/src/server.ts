import { createServer } from 'http';
import expressApp from 'express';
import { Server as SocketIOServer } from 'socket.io';

import config from './config';
import { expressLoader, ioLoader, ormLoader } from './loader';
import PythonSpawner from './ml/pythonSpawner';

class Server {
  public static async start(): Promise<void> {
    const app = expressApp();
    const server = createServer(app);
    const io = new SocketIOServer(server, { cors: { origin: '*' } });

    /** Loaders */
    expressLoader(app);
    ioLoader(io);
    await ormLoader();

    server.listen(config.HTTP_PORT, () => {
      // eslint-disable-next-line no-console
      console.log(`Server running on port ${config.HTTP_PORT}`);
    });

    // 1. train.py - Example of options and thread to train models
    // const uuid = "1aaa41fa-526e-47c6-916c-07906127df3c"
    // const uuid = "b41df80e-4f36-4afa-9428-00939882ff1b"
    // const process = new PythonSpawner('./src/ml/AlphaZero_Gomoku','train.py', ['-u',uuid]);
    // const process = new PythonSpawner('./src/ml/AlphaZero_Gomoku','train.py', ['-u',uuid,'-r']);
    // const process = new PythonSpawner('./src/ml/AlphaZero_Gomoku','train.py', ['-u',uuid,'-r','-c']); //Force cpu use

    // 2. game.py - Example of options and thread to run model-to-model battle
    // For more information of the configuration, look in battle_example.json
    // .json file may be subjected for change
    const gameOptions = ['-g', '../battle/battle_example.json', '-R', '1'];
    // var game_options = ['-g', './data/battle_example.json', '-R', '10'];
    const process = new PythonSpawner(
      './src/ml/AlphaZero_Gomoku',
      'game.py',
      gameOptions
    );

    // 3. human_play.py - Example of options and thread to play with the model
    // const process = new PythonSpawner('./src/ml/AlphaZero_Gomoku','human_play.py', ["-g", "./data/play_example.json"]);

    await process.run().catch(e => console.log(e));
  }
}

export default Server;
