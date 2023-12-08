
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


def infer(classifier, text):
  y_pred=[]
  pred=classifier(text)
  for dict_ in pred:
    if dict_['label']=='POSITIVE':
      y_pred.append(1)
    else:
      y_pred.append(0)
  return y_pred


def test(data_path, model_path):
  
  print('Loading test data ...')
  data = pd.read_csv(data_path)


  print('Loading tokenizer ...')
  ft_tokenizer = AutoTokenizer.from_pretrained(model_path)

  print('Loading model ... ')
  ft_model = AutoModelForSequenceClassification.from_pretrained(model_path)

  print('Start infering ...')
  classifier = pipeline("sentiment-analysis", model=ft_model, tokenizer=ft_tokenizer)

  y_actuals = data.label
  y_predicted = infer(classifier, data.tweet.tolist())
  print(classification_report(y_actuals, y_predicted))

  return


if __name__ == '__main__':
#   test()
#   infer()
  pass
