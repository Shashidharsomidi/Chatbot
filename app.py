import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Configure NLTK and SSL
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Preprocess the data
patterns = []
tags = []
responses = {}

for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

# Tokenize the patterns
nltk.data.path.append(os.path.abspath("nltk_data"))
from nltk.tokenize import word_tokenize
all_words = []
for pattern in patterns:
    all_words.extend(word_tokenize(pattern))

all_words = sorted(set([w.lower() for w in all_words if w.isalnum()]))

# Encode tags
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)

# Vectorize patterns using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='word', token_pattern=r'\b\w+\b')
X = vectorizer.fit_transform(patterns)

# Train the Logistic Regression model
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(X, y)

def chatbot_response(input_text):
    # Preprocess and vectorize user input
    input_vector = vectorizer.transform([input_text])

    # Predict the tag
    tag_idx = clf.predict(input_vector)[0]
    tag = label_encoder.inverse_transform([tag_idx])[0]

    # Select a random response
    return random.choice(responses[tag])

def main():
    st.title("Enhanced Chatbot with Logistic Regression")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome! Type a message below to start chatting.")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input = st.text_input("You:")
        if user_input:
            response = chatbot_response(user_input)
            st.text_area("Chatbot:", value=response, height=120)

            # Save the conversation
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")

    elif choice == "About":
        st.write("This chatbot uses Logistic Regression with TF-IDF for efficient intent recognition. It processes intents effectively and is designed to handle diverse inputs.")

if __name__ == "__main__":
    main()
