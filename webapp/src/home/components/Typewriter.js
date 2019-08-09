import React, { Component } from "react";

export default class Typewriter extends Component {
  state = {
    current: 0
  };

  componentDidMount() {
    const { children, timeMs } = this.props;

    this.interval = setInterval(() => {
      console.log("update");
      const { current } = this.state;
      if (current === children.length) {
        clearInterval(this.interval);
      }
      this.setState({ current: current + 1 });
    }, Math.floor(timeMs / children.length));
  }

  componentWillUnmount() {
    clearInterval(this.interval);
  }

  render() {
    const { children } = this.props;
    const { current } = this.props;
    return <div>{children.substr(0, current)}</div>;
  }
}
