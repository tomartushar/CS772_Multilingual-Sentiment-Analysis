## This file contains code for connecting the model in the background to the GUI which is web interface (Anvil).


import anvil.server
from transformers import pipeline
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# The key given here changes every session of the web interface.
anvil.server.connect(os.getenv('KEY'))

@anvil.server.callable
def predict_sentiment(input_text,selected_model):
  if selected_model=="DistilBERT":
    # save_directory = "/content/drive/MyDrive/DLNLP_Project/finetuned_models/distilbert-base-uncased-finetuned-sst-2-english-1"
    model_path = os.path.join('models', 'DistilBert')
    ft_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    ft_tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier = pipeline("sentiment-analysis", model=ft_model, tokenizer=ft_tokenizer)
    result = classifier([input_text])
    sentiment = result[0]['label']
    score = result[0]['score']
    print(input_text)
    print(sentiment)
    return sentiment, score
  
  elif selected_model=="mBERT":
    # save_directory = "/content/drive/MyDrive/DLNLP_Project/finetuned_models/mbert"
    model_path = os.path.join('models', 'mBERT')
    ft_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    ft_tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier = pipeline("sentiment-analysis", model=ft_model, tokenizer=ft_tokenizer)
    result = classifier([input_text])
    star_value = result[0]['label']
    score = result[0]['score']
    if star_value=='1 star':
      sentiment='NEGATIVE'
    else:
      sentiment='POSITIVE'
    print(input_text)
    print(sentiment)
    return sentiment, score

# This line keeps the background process running and waiting for any input given from the GUI.

if __name__=='__main__':
  anvil.server.wait_forever()