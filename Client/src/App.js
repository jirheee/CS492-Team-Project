import React from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";

import { HomePage, BattlePage, ModelPage } from "./Pages";

export default function App() {
  return (
    <Router>
      <Switch>
        <Route path="/model" component={ModelPage} />
        <Route path="/battle" component={BattlePage} />
        <Route path="/" component={HomePage} />
      </Switch>
    </Router>
  );
}
