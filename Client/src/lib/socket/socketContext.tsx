import { createContext } from 'react';
import { SocketContextInterface } from './types';

const SocketContext = createContext<SocketContextInterface>({
  getSocket: () => null,
  getConnected: () => false,
  getError: () => undefined,
  subscribeEvent: () => undefined,
  getEventMsg: () => ''
});

export default SocketContext;
