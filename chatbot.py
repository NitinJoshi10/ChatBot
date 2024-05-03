# Import necessary libraries
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Download NLTK resources required for text processing
nltk.download('punkt')
nltk.download('wordnet')

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data from a JSON file
data = json.loads(open('data/data.json').read())

# Load preprocessed data and trained model
words = pickle.load(open('word_list.pkl', 'rb'))   # Load preprocessed words data
classes = pickle.load(open('tag_list.pkl', 'rb'))  # Load preprocessed classes data
model = load_model('chatbot_model.h5')             # Load pre-trained model

# Function to clean up sentence by tokenizing and lemmatizing words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())  # Tokenize the sentence
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Lemmatize the words
    return sentence_words

# Function to create bag of words representation of a sentence
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

# Function to predict the class (intent) of a given sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)  # Create bag of words for the sentence
    res = model.predict(np.array([bow]))[0]  # Predict probabilities of each class
    ERROR_THRESHOLD = 0.25
    # Filter results based on threshold, sort by probability, and format into list
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'data': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Function to get a response based on the predicted intent
def get_response(list, data_json):
    tag = list[0]['data']  # Extract the predicted intent
    list_of_data = data_json['data']  # Get the list of all intents
    for i in list_of_data:
        if i['tag'] == tag:  # Find the intent matching the predicted tag
            result = random.choice(i['responses'])  # Randomly select a response from the intent
            break
    return result

# Main code execution starts here
print("ChatBot is working!. Write your query below.")

# Continuous loop to accept user input and provide responses
while True:
    message = input("")  # Get user input
    ints = predict_class(message)  # Predict the intent of the user input
    res = get_response(ints, data)  # Get a response based on the predicted intent
    print(res)  # Print the response