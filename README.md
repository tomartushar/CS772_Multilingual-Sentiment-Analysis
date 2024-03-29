# Multilingual Sentiment Analysis
A multilingual analysis system that can be used for sentiment classification tasks over four different languages `English`, `Hindi`, `Marathi`, and `Gujarati`.

## Setup
* Input: Any sentence in `English`, `Hindi`, `Marathi`, and `Gujarati`.
* Output: Sentiment Class (`POSITIVE` or `NEGATIVE`)
* The user has been provided a very easy-to-use GUI to use this system as a web interface.

## Training
The model has been finetuned on target languages data, generated with the help of Google Translation API.
```
Dataset: Sentiment140
```
Two different kinds of model has been tested:
1. DistillBert
2. mBERT

## Results
| Model           | Class | Precision | Recall | f1-Score |
|-----------------|-------|-----------|--------|----------|
| DistilBERT      |   0   |    0.77   |  0.82  |   0.80   |
|                 |   1   |    0.81   |  0.76  |   0.78   |
| mBERT           |   0   |    0.71   |  0.74  |   0.73   |
|                 |   1   |    0.74   |  0.71  |   0.72   |
