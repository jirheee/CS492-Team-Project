import * as React from 'react';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import { ChakraProvider, theme } from '@chakra-ui/react';
import { ColorModeSwitcher } from './ColorModeSwitcher';
import { MainIndexPage, AgentIndexPage, BattleIndexPage } from './pages';
import { SocketProvider } from './lib/socket';

export const App = () => {
  return (
    <SocketProvider url="localhost:5000">
      <ChakraProvider theme={theme}>
        {/* <ColorModeSwitcher /> */}
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
    </SocketProvider>
  );
};
