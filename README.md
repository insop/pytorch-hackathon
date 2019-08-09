# Pytorch-Hackathon
## Audimext: An approach to aid hearing impaired people

Team Members: Aakash NS, Insop Song, Himanshu Raj, Viraj Deshwal, Paridhi Singh

## 1. Introduction

This project is an implementation inspired by a social cause to help hearing impaired people, not only in understanding and interpreting the speech by other humans around but also to analyse the noise in their background. Our model give them the option to grasp all the information like any other person by casting the background noise in the form of image/gifs at the same time displaying the text of the spoken conversation by other people. The application of this model can also be extended in transportation where existing models fail to depict the scene, i.e., convey the driver who is hearing impaired to know if there is a car honk at him. This could otherwise be a very dangerous situation.

## 2. Datasets and framework

We built the project on Pytorch and it's high end API - Fastai. We have used following datasets :
1. Urban Sound Dataset (8K)  https://urbansounddataset.weebly.com/urbansound8k.html
2. AN4 (800) http://www.speech.cs.cmu.edu/databases/an4/

## 3. Model Design

We created a hybrid model which takes input in any wav form and gives the output in the form of Gif (for background) and text (for speech). We used two different pre-trained networks to classify various background and foreground sounds and transcribe the foreground speech, by converting the .wav files to its equivalent spectrogram using FFT and feeding them to the respective networks.

1. ResNet50: Our input data, in the .wav form is initially fed into the ResNet50 model to help us segregate the 10 different background classes with 1 foreground (speech) class. 

2. Transformer based ASR (FairSeq): We feed the output from the ResNet50 model if foreground class is detected, to the Transformer based ASR to convert speech to text.  (https://arxiv.org/abs/1904.11660)

## 4. APIs
We prepared a web service using Flask and Rest API, that completes our project pipeline by stiching the networks, feeding them with the desired inputs and collect the outputs to present them in the respective visual formats. Our User Interface gives the user flexibility to click and play. 


## 5. External Software
- Pytorch Fairseq: https://github.com/pytorch/fairseq
- Pytorch Audio: https://github.com/pytorch/audio
- Fastai: https://github.com/fastai
