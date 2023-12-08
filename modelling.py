import os
import pandas as pd
import numpy as np
from datasets import load_metric
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments


class TweetDataset(Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels
  
  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key,val in self.encodings.items()}
    item["labels"] = torch.tensor(self.labels[idx])
    return item
  
  def __len__(self):
    return len(self.labels)


def load_dataset(tokenizer, data_path):

  print('Loading data ...')
  indic_df = pd.read_csv(data_path)

  training_texts = indic_df.tweet.tolist()
  training_labels = indic_df.lable.tolist()

  train_texts, val_texts, train_labels, val_labels = train_test_split(training_texts, training_labels, test_size=0.2, shuffle=True)

  train_encodings = tokenizer(train_texts, truncation=True, padding=True)
  val_encodings = tokenizer(val_texts, truncation=True, padding=True)

  train_dataset = TweetDataset(train_encodings, train_labels)
  val_dataset = TweetDataset(val_encodings, val_labels)

  return train_dataset, val_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = load_metric("accuracy")
    return metric.compute(predictions=predictions, references=labels)


def get_tokenizer(model_name):
  print('Loading tokenizer ...')
  return AutoTokenizer.from_pretrained(model_name)


def get_model(model_name):
  print('Loading model ...')
  return AutoModelForSequenceClassification.from_pretrained(model_name)


def train(tokenizer, model, train_dataset, val_dataset, model_path):


  print('Start training ...')

  training_args = TrainingArguments(
    output_dir='/model/results',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=2000,
    eval_steps=2000,
    save_total_limit=2,
    load_best_model_at_end=True
  )


  trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics=compute_metrics
  )

  print(trainer.train())

  tokenizer.save_pretrained(model_path)
  model.save_pretrained(model_path)



def run(model_name, data_path = 'data/train.csv', \
          model_dir = 'models'):
  
  tokenizer, model = get_tokenizer(model_name), get_model(model_name)

  train_dataset, val_dataset = load_dataset(tokenizer, data_path)

  if model_name == 'DistilBert':
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model_path = os.path.join(model_dir, 'DistilBert')
  elif model_name == 'mBERT':
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment" 
    model_path = os.path.join(model_dir, 'mBERT')
  else:
    raise TypeError('Choose model_name either "DistilBert" or "mBERT"')

  train(tokenizer, model, train_dataset, val_dataset, model_path) 


if __name__ == '__main__':
  # run()
  pass

