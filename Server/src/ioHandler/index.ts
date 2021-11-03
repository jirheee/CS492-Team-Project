import { Server as SocketIOServer } from 'socket.io';

export default (io: SocketIOServer) => {
  io.on('connection', socket => {
    console.log('connection');

    const emitResponse = () => {
      socket.emit('response', { time: new Date().toISOString() });
      setTimeout(emitResponse, 2000);
    };

    emitResponse();

    socket.on('disconnect', () => {
      console.log('disconnected');
      socket.disconnect();
      io.emit('user', 'disconnected');
    });

    socket.on('handshake', msg => {
      console.log(msg);
      socket.emit('handshake', msg + msg);
    });
  });
};
