import { Server as SocketIOServer } from 'socket.io';
import ioHandler from '../ioHandler';

export default (io: SocketIOServer) => {
  // TODO: Add io middlewares for Authentication

  ioHandler(io);
};
