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

    const uuid = "1aaa41fa-526e-47c6-916c-07906127df3c"
    const process = new PythonSpawner('./src/ml/AlphaZero_Gomoku','train.py', ['-u',uuid]);
    // const process = new PythonSpawner('./src/ml/AlphaZero_Gomoku','train.py', ['-u',uuid,'-r']);

    //var game_options = ["-g", "./data/battle_example.json"]
    // var game_options = ["-g", "./data/battle_example.json", "-R", "10"]
    // const process = new PythonSpawner('./src/ml/AlphaZero_Gomoku','game.py', game_options);
    
    // const process = new PythonSpawner('./src/ml/AlphaZero_Gomoku','human_play.py', ["-g", "./data/play_example.json"]);
    await process.run();
  }
}

export default Server;
