
import pandas as pd
import numpy as np
from googletrans import Translator


def translation(translator, target, text):
  translated_text=translator.translate(text, dest=target)
  return translated_text


def convertor(data, bound = 40000):
    translator=Translator()
    for i in range(bound, bound+bound):
        data['tweet'][i] = translation(translator, 'hi', data['tweet'][i]).text

    for i in range(2*bound, 3*bound):
        data['tweet'][i] = translation(translator, 'gu', data['tweet'][i]).text

    for i in range(3*bound, data.shape[0]):
        data['tweet'][i] = translation(translator, 'mr', data['tweet'][i]).text

    return data

def prepare_data(data_path = 'data/data.csv'):

    print('Preparing data ...')
    df = pd.read_csv(data_path, encoding='latin-1')
    df_sentiment = df[['tweet','label']]

    df_sentiment = df_sentiment.sample(frac=1).reset_index(drop = True)

    df_sentiment = df_sentiment.iloc[:160000]
    df_sentiment = convertor(df_sentiment)

    df_sentiment = df_sentiment.sample(frac=1)

    df_sentiment['label'][df.label == 4] = 1

    df_train = df_sentiment.iloc[:int(0.85*df_sentiment.shape[0])]
    df_test = df_sentiment.iloc[int(0.85*df_sentiment.shape[0]):]

    df_train.to_csv('data/train.csv', index = False)
    df_test.to_csv('data/test.csv', index = False)

