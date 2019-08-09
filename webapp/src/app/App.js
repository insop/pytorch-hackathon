import React from "react";
import { BrowserRouter, Switch, Route } from "react-router-dom";
import HomePage from "home/pages/HomePage";
import ResultPage from "result/pages/ResultPage";

const App = () => (
  <BrowserRouter>
    <Switch>
      <Route path="/result" component={ResultPage} />
      <Route path="" component={HomePage} />
    </Switch>
  </BrowserRouter>
);

export default App;
