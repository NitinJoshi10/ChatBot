# Import necessary libraries
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Load and preprocess data
lemmatizer = WordNetLemmatizer()
data = json.loads(open('data/data.json').read())

# Initialize lists to store words, tags, and documents
word_list = []
tag_list = []
document_list = []
ignore_letters = ['?', '!', '.', ',']  # Characters to ignore

# Process data to extract words, tags, and documents
for data_item in data['data']:
    for pattern in data_item['patterns']:
        word_list.extend(nltk.word_tokenize(pattern))  # Tokenize patterns into words
        document_list.append((nltk.word_tokenize(pattern), data_item['tag']))  # Store word-tag pairs
        if data_item['tag'] not in tag_list:
            tag_list.append(data_item['tag'])  # Collect unique tags

# Lemmatize and sort word list
word_list = [lemmatizer.lemmatize(word) for word in word_list if word not in ignore_letters]
word_list = sorted(set(word_list))
tag_list = sorted(set(tag_list))

# Save preprocessed data
pickle.dump(word_list, open('word_list.pkl', 'wb'))
pickle.dump(tag_list, open('tag_list.pkl', 'wb'))

# Create training data
training_data = []
output_empty = [0] * len(tag_list)

# Create bag of words representation for each document and one-hot encode tags
for document in document_list:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in word_list:
        bag.append(1) if word in word_patterns else bag.append(0)  # Mark presence of words
    output_row = list(output_empty)
    output_row[tag_list.index(document[1])] = 1  # One-hot encode the tag
    training_data.append(bag + output_row)

# Shuffle and convert training data to numpy array
random.shuffle(training_data)
training_data = np.array(training_data)

# Split data into features (X) and labels (Y)
train_X = training_data[:, :len(word_list)]
train_Y = training_data[:, len(word_list):]

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_X[0]),), activation='relu'),  # Input layer
    tf.keras.layers.Dropout(0.5),  # Dropout layer to prevent overfitting
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
    tf.keras.layers.Dropout(0.5),  # Dropout layer
    tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer
    tf.keras.layers.Dropout(0.5),  # Dropout layer
    tf.keras.layers.Dense(len(train_Y[0]), activation='softmax')  # Output layer
])

# Compile the model with optimizer and loss function
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
history = model.fit(train_X, train_Y, epochs=250, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5', history)