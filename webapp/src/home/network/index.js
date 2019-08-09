const API_URL = "http://35.233.181.217:8090";

export const uploadAudio = file => {
  const formData = new FormData();

  formData.append("audio", file);

  const options = {
    method: "POST",
    body: formData
  };

  return fetch(API_URL + "/transcribe", options).then(res => res.json());
};
