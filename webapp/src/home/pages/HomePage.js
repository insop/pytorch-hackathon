import React, { Component } from "react";
import { Container, Button } from "react-bootstrap";
import { FiUpload } from "react-icons/fi";
import { uploadAudio } from "../network";
import { Loader } from "react-overlay-loader";
import "react-overlay-loader/styles.css";
import ShowResult from "home/components/ShowResult";

class UploadButton extends Component {
  state = {
    loading: false,
    finished: false,
    audioSrc: null
  };

  handleClick = () => {
    this.fileInput.click();
  };

  submitAudio = file => {
    this.setState({ audioSrc: URL.createObjectURL(file) });
    uploadAudio(file).then(({ success, result }) => {
      if (success) {
        console.log(result);
        this.setState({ loading: false, finished: true, apiResult: result });
      } else {
        alert("Something went wrong! :(");
        window.reload();
      }
    });
  };

  handleChange = e => {
    console.log(e.target.files);
    const formInput = e.target;
    if (formInput.files.length > 0) {
      this.setState({ loading: true }, () =>
        this.submitAudio(formInput.files[0])
      );
    }
  };

  render() {
    const { loading, finished, audioSrc, apiResult } = this.state;
    if (loading) {
      return <Loader loading fullPage text="Understating what's going on.." />;
    }

    if (finished) {
      return <ShowResult audioSrc={audioSrc} apiResult={apiResult} />;
    }

    return (
      <>
        <input
          type="file"
          id="audio"
          style={{ visibility: "hidden" }}
          ref={el => (this.fileInput = el)}
          onChange={this.handleChange}
        />
        <Button
          style={{
            width: 260,
            height: 120,
            borderRadius: 20,
            // backgroundColor: "#2E7D32",
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "center",
            color: "white",
            fontSize: 48
          }}
          onClick={this.handleClick}
        >
          <div>
            <FiUpload style={{ color: "white", marginBottom: 10 }} />
            &nbsp;
          </div>
          <div>Upload</div>
        </Button>
        <img
          src="https://i.imgur.com/NFS20Om.png"
          style={{ marginTop: 72, height: 200, width: 200 }}
          alt=""
        />
        <div style={{ fontSize: 12 }}>
          Please enable "Insecure scripts" as our REST API is on HTTP.
        </div>
      </>
    );
  }
}

const HomePage = () => (
  <Container>
    <h1
      style={{
        fontSize: 48,
        textAlign: "center",
        marginTop: 32,
        marginBottom: 64
      }}
    >
      Audio to Image
    </h1>
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center"
      }}
    >
      <UploadButton />
    </div>
  </Container>
);

export default HomePage;
