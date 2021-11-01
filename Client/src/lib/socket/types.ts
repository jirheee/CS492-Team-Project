/* eslint-disable no-shadow */
import { ManagerOptions, Socket, SocketOptions } from 'socket.io-client';

export type SocketContextInterface = {
  getSocket: () => Socket | null;
  getConnected: () => boolean;
  getError: () => any;
  subscribeEvent: (event: string) => void;
  getEventMsg: (event: string) => string;
};

export type SocketOptionType = Partial<SocketOptions & ManagerOptions> & {
  useMock: boolean;
};
