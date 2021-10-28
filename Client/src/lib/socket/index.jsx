import React, { useContext, useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";

const CONNECTED = "connected";
const DISCONNECTED = "disconnected";

const SocketContext = React.createContext({
  createSocket: () => {},
  getSocket: () => {},
  getState: () => DISCONNECTED,
  getError: () => {},
  setError: () => {},
  getEventMsg: () => {},
  setEventMsg: () => {},
  registerEvent: () => {}
});

function SocketProvider({ children }) {
  const socket = useRef(null);
  const [sockState, setSockState] = useState(DISCONNECTED);
  const [error, setError] = useState(null);
  const [eventMsgRecord, setEventMsgRecord] = useState({});

  const createSocket = (url, socketOptions) => {
    if (socket.current) {
      socket.current.connect();
      return { socket };
    }

    const _socket = new io(url, socketOptions);
    socket.current = _socket;

    _socket.on("error", e => {
      setError(e);
    });
    _socket.on("connect", () => {
      setSockState(CONNECTED);
    });
    _socket.on("disconnect", () => {
      setSockState(DISCONNECTED);
    });

    return { socket };
  };
  const getSocket = () => socket.current;
  const getError = () => error;
  const getState = () => sockState;

  const getEventMsg = (event = "") => eventMsgRecord[event];
  const setEventMsg = (event, msg) =>
    setEventMsgRecord(record => {
      record[event] = msg;
    });

  const registerEvent = event => {
    if (socket.current) {
      socket.current.on(event, msg => {
        setEventMsg(msg);
      });
    }
  };

  return (
    <SocketContext.Provider
      value={{
        createSocket,
        getSocket,
        getError,
        setError,
        getState,
        getEventMsg,
        setEventMsg,
        registerEvent
      }}
    >
      {children}
    </SocketContext.Provider>
  );
}

function useSocket(url = "", socketOptions) {
  console.log("useSocket");
  const { createSocket, getState, getError } = useContext(SocketContext);

  const sockState = getState();
  const error = getError();

  const [connected, setConnected] = useState(false);
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    console.log("change socket state");
    switch (sockState) {
      case CONNECTED: {
        setConnected(true);
        break;
      }
      case DISCONNECTED: {
        setConnected(false);
        break;
      }
      default: {
        throw new Error("invalid Socket State");
      }
    }
  }, [sockState]);

  useEffect(() => {
    console.log("creating socket");
    const { socket } = createSocket(url, socketOptions);
    setSocket(socket);
    return () => {
      console.log("destroy socket");
    };
  }, [createSocket, socketOptions, url]);

  return { socket, error, connected };
}

function useSocketEvent(event) {
  const { registerEvent, getEventMsg, getSocket } = useContext(SocketContext);

  const socket = getSocket();
  const msg = getEventMsg();

  useEffect(() => {
    registerEvent(event);
  }, [event, registerEvent]);

  return { msg, socket };
}

export { SocketProvider, useSocket, useSocketEvent };
