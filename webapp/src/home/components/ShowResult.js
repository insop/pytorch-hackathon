import React, { Component, PureComponent } from "react";
import AudioPlayer from "./AudioPlayer";
// import Typewriter from "./Typewriter";
// import { ActivityClasses } from "../utils";

class Gif extends PureComponent {
  makeGifSrc = cls => {
    return `./${cls}_${Math.random() > 0.5 ? "1" : "2"}.gif`;
  };
  render() {
    const { cls } = this.props;
    return <img src={this.makeGifSrc(cls)} alt={cls} style={{ height: 240 }} />;
  }
}

const Transcription = ({ children }) => (
  <div
    style={{
      marginTop: 64,
      fontFamily: "Georgia",
      fontStyle: "italic",
      fontSize: 20,
      textAlign: "center"
    }}
  >
    <div>{children}</div>
  </div>
);

export default class ShowResult extends Component {
  state = {
    duration: 0,
    idx: null
  };

  componentDidMount() {}

  handleListen = t => {
    const idx = Math.floor(t / 4);
    this.setState({ idx });
  };

  handlePlay = () => {
    this.handleListen(0.1);
  };

  renderText = () => {
    console.log("renderText", this.props);
    const { idx } = this.state;
    if (idx === null) {
      return null;
    }
    const { apiResult } = this.props;
    const { cls, transcription } = apiResult[idx];

    if (cls === "speech") {
      return <Transcription>{transcription}</Transcription>;
    }
    return null;
  };

  renderGif = () => {
    const { idx } = this.state;
    if (idx === null) {
      return null;
    }
    const { apiResult } = this.props;
    const { cls } = apiResult[idx];
    if (cls !== "speech") {
      return (
        <div>
          <Gif cls={cls} />
        </div>
      );
    }
    return null;
  };

  render() {
    const { audioSrc } = this.props;
    return (
      <div>
        <div
          style={{
            alignSelf: "stretch",
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          }}
        >
          <AudioPlayer
            src={audioSrc}
            controls
            listenInterval={500}
            onLoadedMetadata={this.handleMetadata}
            onListen={this.handleListen}
            onEnded={this.handleEnd}
            onPlay={this.handlePlay}
          />
        </div>

        {this.renderText()}
        {this.renderGif()}
      </div>
    );
  }
}
