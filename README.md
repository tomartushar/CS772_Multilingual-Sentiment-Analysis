# Multilingual Sentiment Analysis

A multilingual analysis system designed for sentiment classification across four different languages: `English`, `Hindi`, `Marathi`, and `Gujarati`.

## Setup
- **Input**: Any sentence in `English`, `Hindi`, `Marathi`, and `Gujarati`.
- **Output**: Sentiment Class (`POSITIVE` or `NEGATIVE`)
- The system features an intuitive GUI for seamless web interface interaction.

## Training
The model has undergone fine-tuning on target language data, facilitated by the Google Translation API.

**Dataset**: Sentiment140

Two distinct model architectures have been evaluated:
1. DistilBERT
2. mBERT

## Results
| Model           | Class | Precision | Recall | F1-Score |
|-----------------|-------|-----------|--------|----------|
| DistilBERT      |   0   |    0.77   |  0.82  |   0.80   |
|                 |   1   |    0.81   |  0.76  |   0.78   |
| mBERT           |   0   |    0.71   |  0.74  |   0.73   |
|                 |   1   |    0.74   |  0.71  |   0.72   |

