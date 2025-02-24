from django.shortcuts import render
from django.http import HttpResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import numpy as np

def home(request):
    return render(request,'home.html')

def filter_hotels_by_city_and_price(dataset, city, min_price, max_price):
    filtered_hotels = dataset[(dataset['city'] == city) & 
                              (dataset['price'] >= min_price)&
                              (dataset['price'] <= max_price)]
    return filtered_hotels

def top_5_hotels_sorted(dataset):
    sorted_dataset = dataset.sort_values(by='hotel_rating', ascending=False)
    top_5 = sorted_dataset.head(5)
    return top_5

def top_3_lowest_priced_hotels(dataset):
    sorted_dataset = dataset.sort_values(by='price', ascending=True)
    top_3 = sorted_dataset.head(3)
    return top_3

def get_prediction(t):
    words = []

    for word in t.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        words.append(word)

    proc = " ".join(words)

    # load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    # sentiment analysis
    encoded = tokenizer(proc, return_tensors='pt')
    output = model(**encoded)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    print(scores)

    return scores

def submit_form(request):
    if request.method == 'POST':
        city = request.POST.get('input_text', '')
        rangeemin = int(request.POST.get('input_text1', ''))
        rangeemax = int(request.POST.get('input_text2', ''))

        if city == '':
            return HttpResponse("Error: city not entered.")

        if not rangeemin:
            return HttpResponse("Error: range min not entered.")

        if not rangeemax:
            return HttpResponse("Error: range max not entered.")

        print(city)
        print(rangeemin)
        print(rangeemax)
        # Process input_text and uploaded_file here

        data = pd.read_csv("C:\Users\Admin\Downloads\TravelProject.V.01\travel\travel\senti\webapp\filtered_hotel.csv")

        fd = filter_hotels_by_city_and_price(data, city, rangeemin, rangeemax)
        t5 = top_5_hotels_sorted(fd)
        t3 = top_3_lowest_priced_hotels(t5)

        for index, row in t3.iterrows():
            prediction = get_prediction(row['reviews'])  # Assuming you're passing the third review to get_prediction()
            print(prediction)
            points = round(float(row['hotel_rating'] * prediction[2] / row['price']), 2)

            t3.at[index, 'points'] = points

        max_points_row_index = t3['points'].idxmax()
        max_points_row = t3.loc[max_points_row_index]

        max_points_dict = {
            "id": max_points_row['id'],
            "hotel_name": max_points_row['hotel_name'],
            "hotel_rating": max_points_row['hotel_rating'],
            "hotel_experience": max_points_row['hotel_experience'],
            "amenities": max_points_row['amenities'],
            "address": max_points_row['address'],
            "country": max_points_row['country'],
            "location": max_points_row['location'],
            "price": max_points_row['price'],
            "city": max_points_row['city'],
            "reviews": max_points_row['reviews'],
            "points": max_points_row['points']
        }

        print(max_points_dict)
        return render(request,'output.html',max_points_dict)

    else:
        return HttpResponse("Error: Form submission failed.")
