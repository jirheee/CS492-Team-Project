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

export const App = () => {
  return (
    <ChakraProvider theme={theme}>
      {/* <ColorModeSwitcher /> */}
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
