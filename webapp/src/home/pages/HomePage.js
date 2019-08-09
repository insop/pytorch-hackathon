import React from "react";
import { Link } from "react-router-dom";

const HomePage = () => (
  <div>
    This is the homepage. <Link to="/result">Result</Link>
  </div>
);

export default HomePage;
