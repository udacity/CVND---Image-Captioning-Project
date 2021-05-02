# Image Captioning Project


This project demonstrates the usage of a combined CNN-RNN to generate captions from images. 
It was completed as part of Udacity's Computer Vision Nano-degree and is inspired by
the paper - "Show and Tell : A Neural Image Caption Generator - ", but is not meant at all to reproduce their results. 

An overview and example of the results. 


# Setup & Usage

1. Download data
2. Create and configure environment
3. Run Training ( Can skip this step ). Notebook 1. 
4. Evaluate on Test Data.               Notebook 2.


# How & Why it Works 

![Encoder Decoder Model](images/encoder-decoder.png)

## Model
The Encoder CNN is a pretrained ResNet50 with fixed ( non-trainable ) parameters. 
The Decoder RNN is composed of X layers of LSTM cells. 


## Training
The dataset is composed of ... etc. & this is how the network is trained. 


## Inference
I used a simple sampling based approach to caption generation. 
In this the RNN is fed in the encoded image after passing through the CNN. 
After that, the words are sampled one by one through the RNN until the maximum length of X is reached. 





