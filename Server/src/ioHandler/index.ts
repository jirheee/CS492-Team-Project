import { Server as SocketIOServer } from 'socket.io';

export default (io: SocketIOServer) => {
  io.on('connection', socket => {
    console.log('connection');
    socket.on('disconnect', () => {
      io.emit('user', 'disconnected');
    });

    socket.on('handshake', msg => {
      console.log(msg);
      socket.emit('handshake', msg + msg);
    });
  });
};
