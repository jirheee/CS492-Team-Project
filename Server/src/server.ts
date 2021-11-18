import { createServer } from 'http';
import expressApp from 'express';
import { Server as SocketIOServer } from 'socket.io';

import config from './config';
import { expressLoader, ioLoader, ormLoader } from './loader';
// import PythonSpawner from './ml/pythonSpawner';

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

    // const process = new PythonSpawner('./dummy.py');
    // await process.run();
  }
}

export default Server;
