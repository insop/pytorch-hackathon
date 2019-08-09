import React, { Component } from "react";
import { Link } from "react-router-dom";
import AudioPlayer from "../components/AudioPlayer";
import CrossfadeImage from "../components/CrossfadeImage";
import { Container } from "react-bootstrap";

const SAMPLE_CAPTION = `Lorem Ipsum is simply dummy text of the printing 
and typesetting industry. Lorem Ipsum has been the industry's standard 
dummy text ever since the 1500s, when an unknown printer took a galley 
of type and scrambled it to make a type specimen book. It has survived 
not only five centuries, but also the leap into electronic typesetting, 
remaining essentially unchanged.`;

const GIFS = [
  {
    label: "dog_bark",
    image: "https://i.giphy.com/media/939RsumhGX9n4amBlN/giphy.webp"
  },
  {
    label: "car driving",
    image: "https://i.giphy.com/media/9F2VWRiJypeBq/giphy.webp"
  },
  {
    label: "rain",
    image: "https://i.giphy.com/media/Mgq7EMQUrhcvC/giphy.webp"
  },
  {
    label: "dog_bark",
    image: "https://i.giphy.com/media/939RsumhGX9n4amBlN/giphy.webp"
  },
  {
    label: "car driving",
    image: "https://i.giphy.com/media/9F2VWRiJypeBq/giphy.webp"
  },
  {
    label: "rain",
    image: "https://i.giphy.com/media/Mgq7EMQUrhcvC/giphy.webp"
  },
  {
    label: "car driving",
    image: "https://i.giphy.com/media/9F2VWRiJypeBq/giphy.webp"
  }
];

export default class ResultPage extends Component {
  state = {
    loading: true,
    fullText: SAMPLE_CAPTION,
    duration: 0,
    text: "",
    gifs: GIFS,
    currentGif: null
  };

  handleMetadata = e => {
    this.setState({
      loading: false,
      duration: e.target.duration
    });
  };

  handleListen = t => {
    // set the text
    const { duration, fullText, gifs } = this.state;
    const end = Math.ceil(((t + 1) * fullText.length) / duration);
    this.setState({ text: fullText.substr(0, end) });

    // set the Imagery
    let thres = Math.ceil(t);
    if (thres % 2 === 1) {
      thres--;
    }
    thres = thres / 2;
    thres = Math.min(gifs.length - 1, thres);
    console.log("Image: ", thres);
    this.setState({ currentGif: gifs[thres] });
  };

  handleEnd = () => {
    this.setState({ text: this.state.fullText });
  };

  handlePlay = () => {
    this.handleListen(0);
  };

  render() {
    const { text, currentGif } = this.state;
    return (
      <Container>
        <h1>Audio Imagery Result</h1>
        <div>
          Go <Link to="/">Home</Link>
        </div>
        <AudioPlayer
          src="https://www.nch.com.au/acm/11k16bitpcm.wav"
          controls
          listenInterval={1000}
          onLoadedMetadata={this.handleMetadata}
          onListen={this.handleListen}
          onEnded={this.handleEnd}
          onPlay={this.handlePlay}
        />
        <div>
          <h3>Transcription</h3>
          <div style={{ minHeight: 120 }}>{text}</div>
        </div>
        <div>
          <h3>Visual Imagery</h3>
          <div>
            {currentGif && (
              <CrossfadeImage
                style={{ height: 320 }}
                src={currentGif.image}
                alt={currentGif.label}
              />
            )}
          </div>
        </div>
      </Container>
    );
  }
}
