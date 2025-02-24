from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import PyPDF2
import re


def get_prediction(tweet):
    
    
    tweet_words = []

    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)

    # load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)


    labels = ['Negative', 'Neutral', 'Positive']

    # sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    print(scores)

    return scores

def extract(pdf_file: str)->str:
    with open (pdf_file,'rb')as pdf:
        reader=PyPDF2.PdfReader(pdf,strict=False)
        pdf_text=[]

        for page in reader.pages:
            content= page.extract_text()
            pdf_text.append(content)

        return pdf_text


tweet = "@MehranShakarami today's cold @ home 😀 https://mehranshakarami.com"
tweet = 'Not really Good😒'
get_prediction(tweet)


extracted_text=extract('example.pdf')
for text in extracted_text:
    # split_message=re.split(r'\s+|[,;?!.-]\s*',text.lower())
    print(text)