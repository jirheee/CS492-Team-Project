import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import { ChakraProvider, theme, Heading } from '@chakra-ui/react';
// import { ColorModeSwitcher } from './ColorModeSwitcher';
import { MainIndexPage, AgentIndexPage, BattleIndexPage } from './pages';
import { SocketProvider, useSocket } from './lib/socket';
import { useEffect, useState } from 'react';

import Request from './lib/api/request';

export const App = () => {
  const [text, setText] = useState('');

  useEffect(() => {
    console.log('Rendered');
    Request.get('/status').then(data => {
      console.log(data.data.data);
      setText(data.data.data);
    });
  }, []);

  const { socket, connected } = useSocket();
  const [socketResponseText, setSocketText] = useState('');

  useEffect(() => {
    socket?.on('response', (data: Record<string, string>) => {
      console.log(data);
      setSocketText(data['time']);
    });
  }, [socket, connected]);

  return (
    <ChakraProvider theme={theme}>
      {/* <ColorModeSwitcher /> */}
      <Heading>{`http response: ${text}`}</Heading>
      <Heading>{`socket response: ${socketResponseText} socket connected: ${
        connected ? 'connected' : 'disconnected'
      }`}</Heading>
      <Router>
        <Switch>
          <Route exact path="/">
            <MainIndexPage />
          </Route>
          <Route path="/agent">
            <AgentIndexPage />
          </Route>
          <Route path="/battle">
            <BattleIndexPage />
          </Route>
        </Switch>
      </Router>
    </ChakraProvider>
  );
};
