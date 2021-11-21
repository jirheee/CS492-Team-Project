import {
  BrowserRouter as Router,
  Switch,
  Route,
  Redirect
} from 'react-router-dom';
import { ChakraProvider, theme, Box } from '@chakra-ui/react';
// import { ColorModeSwitcher } from './ColorModeSwitcher';
import {
  MainIndexPage,
  AgentIndexPage,
  BattleIndexPage,
  BattlePage,
  AgentCreatePage,
  AgentManagementPage
} from './pages';
import { useSocket } from './lib/socket';
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
      {/* <Heading>{`http response: ${text}`}</Heading>
      <Heading>{`socket response: ${socketResponseText} socket connected: ${
        connected ? 'connected' : 'disconnected'
      }`}</Heading> */}
      <Box w="100vw" h="100vh">
        <Router>
          <Switch>
            <Route exact path="/" component={MainIndexPage} />
            <Route exact path="/agent" component={AgentIndexPage} />
            <Route exact path="/agent/create" component={AgentCreatePage} />
            <Route exact path="/agent/manage" component={AgentManagementPage} />
            <Route exact path="/battle" component={BattleIndexPage} />
            <Route exact path="/battle" component={BattleIndexPage} />
            <Route exact path="/battle/game" component={BattlePage} />
            <Route path="/*">
              <Redirect to="/" />
            </Route>
          </Switch>
        </Router>
      </Box>
    </ChakraProvider>
  );
};
