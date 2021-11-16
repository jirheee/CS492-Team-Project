import {
  BrowserRouter as Router,
  Switch,
  Route,
  Redirect
} from 'react-router-dom';
import { ChakraProvider, theme, Heading } from '@chakra-ui/react';
// import { ColorModeSwitcher } from './ColorModeSwitcher';
import {
  MainIndexPage,
  AgentIndexPage,
  BattleIndexPage,
  BattlePage
} from './pages';
import { useSocket } from './lib/socket';
import { useEffect, useState } from 'react';

import Request from './lib/api/request';
import AgentCreatePage from './pages/agentCreatePage';

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
          <Route exact path="/" render={MainIndexPage} />
          <Route exact path="/agent" render={AgentIndexPage} />
          <Route exact path="/agent/create" render={AgentCreatePage} />
          <Route exact path="/battle" render={BattleIndexPage} />
          <Route exact path="/battle" render={BattleIndexPage} />
          <Route exact path="/battle/game" render={BattlePage} />
          <Route path="/*">
            <Redirect to="/" />
          </Route>
        </Switch>
      </Router>
    </ChakraProvider>
  );
};
