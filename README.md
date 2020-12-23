# Intent-Classification-for-20-in-scope-labels

## Libraries used:
import pandas as pd<br>
import numpy as np<br>
import requests<br>
import urllib.request as request<br>
from sklearn.preprocessing import OneHotEncoder<br>
from nltk.corpus import stopwords<br>
from nltk.tokenize import word_tokenize<br>
from keras.preprocessing.text import Tokenizer<br>
from nltk.stem.lancaster import LancasterStemmer<br>
import json <br>
import random<br>
import re<br>
import string<br>
from nltk.stem.snowball import SnowballStemmer<br>
from keras.models import Sequential,load_model<br>
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout<br>
from keras.callbacks import ModelCheckpoint<br>
from keras.preprocessing.sequence import pad_sequences<br>
import nltk<br>



For this use case, I moved forward with GRU (Gated recurrent units) based LSTM model. 
General Flow: The data is first read from the .json file and then randomly 20 in-scope contents are extracted from the dataset for each label: train, test and validation. These are further loaded into Pandas data frame for further cleaning pre-processing. The text data is first scanned and then stop words, punctuations and special symbols are removed. The remaining data is then vectorized/tokenized based on the tokenizer and then additional padding is provided to keep the input data of same shape. This data is supplied to the model for training and then using the same model the label probability is calculated for the test data.
Simplification made: For just easiness I have changed/replaced the ‘_’ in the label name with the null space.

## Model Architecture:
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 15, 64)            108928    
_________________________________________________________________
bidirectional_1 (Bidirection (None, 512)               657408    
_________________________________________________________________
dense_1 (Dense)              (None, 32)                16416     
_________________________________________________________________
dropout_1 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 20)                660       
=================================================================

## Training-related parameters:
Dropout = 0.5 after every layer
Final activation layer = ‘relu’
Optimizer='Adam'
Batch size=32
Epochs=150

## The performance evaluation metric:
Evaluation metric = accuracy
Final algorithm’s training accuracy = 95.65
Final algorithm’s validation accuracy = 84

## Challenges:
The biggest challenge was to identify 20 labels having least similarity. This would has helped us in modelling better ML model. I tried to implement cosine similarity for the dataset and based on the resultant matrix, I was planning to select last 20 labels as they would be having least similarity. The biggest challenge and future work would have been to implement this matrix. I was not able to do this because this was taking a lot of resources – memory and CPU utilization. I also tried to make use of google collab but then for extra memory it was asking for premium version.

## Future Work:
1) Implementing the above similarit similarity matrix.
2) Designing other ML models to compare with the GRU-LSTM.

# The above code was encaspulated making use of functions. 
