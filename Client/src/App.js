import React from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import { useSocket } from "./lib/socket/";

import { HomePage, BattlePage, ModelPage } from "./Pages";

export default function App() {
  const { connected } = useSocket("localhost:5000");
  return (
    <Router>
      <div>{connected ? "connected" : "disconnected"}</div>
      <Switch>
        <Route exact path="/" component={HomePage} />
        <Route path="/model" component={ModelPage} />
        <Route path="/battle" component={BattlePage} />
      </Switch>
    </Router>
  );
}
