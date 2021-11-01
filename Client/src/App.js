import React from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import { useSocket } from "./lib/socket/";

import { HomePage, BattlePage, ModelPage } from "./Pages";

export default function App() {
  console.log("App");
  const { connected } = useSocket("localhost:5000");
  return (
    <Router>
      {connected ? "hi" : "hello"}
      <Switch>
        <Route path="/model" component={ModelPage} />
        <Route path="/battle" component={BattlePage} />
        <Route path="/" component={HomePage} />
      </Switch>
    </Router>
  );
}
