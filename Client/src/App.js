import React from "react";
import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link
} from "react-router-dom";

import { ConfigurationPage, TrainingPage, HomePage } from "./Pages";

export default function App() {
  return (
    <Router>
      <div>
        <nav>
          <ul>
            <li>
              <Link to="/">Home</Link>
            </li>
            <li>
              <Link to="/config">Configuration Page</Link>
            </li>
            <li>
              <Link to="/training">Training Page</Link>
            </li>
          </ul>
        </nav>

        <Switch>
          <Route path="/config">
            <ConfigurationPage />
          </Route>
          <Route path="/training">
            <TrainingPage />
          </Route>
          <Route path="/">
            <HomePage/>
          </Route>
        </Switch>
      </div>
    </Router>
  );
}