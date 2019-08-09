# asr-demo

We recommend that you use [conda](https://docs.conda.io/en/latest/miniconda.html) to install these dependencies.

What you need to run this demo

- [python3](https://www.python.org/download/releases/3.0/)
- [torchaudio](https://github.com/pytorch/audio/tree/master/torchaudio)
- [pytorch](https://pytorch.org/)
- [librosa](https://librosa.github.io/librosa/)
- [fairseq](https://github.com/pytorch/fairseq) (clone the github repository)

Models:

- [dictionary](https://download.pytorch.org/models/audio/dict.txt)
- [sentence piece model](https://download.pytorch.org/models/audio/spm.model)
- [model](https://download.pytorch.org/models/audio/checkpoint_avg_60_80.pt)

Example command:
Save the dictionary, sentence piece model and model in data

python interactive_asr.py ./data --max-tokens 10000000 --nbest 1 --path ./data/model.pt --beam 40 --task speech_recognition --user-dir ../fairseq/examples/speech_recognition

File based inference:

```
python infer_file.py ./data --max-tokens 10000000 --nbest 1 --path ./data/checkpoint_avg_60_80.pt --beam 40 --task s
peech_recognition --user-dir ../fairseq/examples/speech_recognition
```

# Follow These Instructions

ASR Demo

Make a directory:

```
git clone asr-dmo
conda create -n asr-demo python=3.7
conda activate air-demo
conda install torchaudio -c pytorch
conda install librosa -c conda-forge
pip install sentencepiece

cd ..
git clone https://github.com/pytorch/fairseq
cd fairseq
export CFLAGS='-stdlib=libc++'
pip install --editable .

cd ../asr-demo
mkdir data
cd data
wget https://download.pytorch.org/models/audio/dict.txt
wget https://download.pytorch.org/models/audio/spm.model
wget https://download.pytorch.org/models/audio/checkpoint_avg_60_80.pt
cd ..
```

Get the inference file

```
wget https://gist.githubusercontent.com/aakashns/2b696fe4b03f37a9d7f57cfd06cb7e5b/raw/573d250c51999e9d2d35dbd039b59dc1d7407806/infer_file.py
```

Download a wav file to data

```
cd data
// wget a .wav file
```

Set the path in file

```
// edit infer_file.py to set path
// set use_cuda to false
```

Final inference

```
python infer_file.py ./data --max-tokens 10000000 --nbest 1 --path ./data/checkpoint_avg_60_80.pt --beam 40 --task speech_recognition --user-dir ../fairseq/examples/speech_recognition
```
