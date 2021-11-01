import * as React from 'react';
import { BrowserRouter as Router, Switch, Route, Link } from 'react-router-dom';
import { ChakraProvider, theme } from '@chakra-ui/react';
import { ColorModeSwitcher } from './ColorModeSwitcher';
import { MainIndexPage, AgentIndexPage, BattleIndexPage } from './pages';

export const App = () => {
  return (
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
  );
};
