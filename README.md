# Chatbot Using Machine Learning
This project is a chatbot implementation using Logistic Regression for intent classification and TF-IDF vectorization for text preprocessing. The chatbot predicts intents based on user inputs and provides appropriate responses from a predefined dataset of intents.

## Features

- *Intent Recognition*: Uses Logistic Regression to classify user inputs into intents.
- *TF-IDF Vectorization*: Converts text inputs into numerical features using n-grams.
- *Streamlit Interface*: Provides a simple web interface for user interaction.
- *Conversation History*: Saves conversations to a CSV file for future reference.

## Requirements

- Python 3.7+
- Libraries:
  - streamlit
  - numpy
  - scikit-learn
  - nltk

## Installation

1. Clone the repository:
   bash
   
   git clone https://github.com/Shashidharsomidi/Chatbot.git

   cd chatbot
   
3. Install the required libraries:
   bash
   
   pip install -r requirements.txt
   
5. Download NLTK data:
   python
   
   import nltk
   
   nltk.download('punkt')
   

## Usage

1. Place your intents in a file named intents.json.
2. Run the Streamlit application:
   bash
   
   streamlit run app.py 
4. Interact with the chatbot in the web interface.

## File Structure

- *app.py*: The main application file containing the chatbot logic.
- *intents.json*: A JSON file defining the chatbot's intents and responses.
- *chat_log.csv*: A CSV file to log conversations.

## Intents JSON Format

An example intents.json file:
json
[
  {
    "tag": "greeting",
    "patterns": ["Hi", "Hello", "How are you?"],
    "responses": ["Hello!", "Hi there!", "How can I assist you?"]
  },
  {
    "tag": "goodbye",
    "patterns": ["Bye", "Goodbye", "See you later"],
    "responses": ["Goodbye!", "Take care!"]
  }
]


## How It Works

1. *Preprocessing*:
   - Tokenizes the patterns in the intents file.
   - Converts text into numerical vectors using TF-IDF.
   - Encodes the intent tags into numerical labels.

2. *Model Training*:
   - Trains a Logistic Regression model on the vectorized patterns and encoded tags.

3. *Inference*:
   - Vectorizes user input using the TF-IDF vectorizer.
   - Predicts the intent tag using the trained Logistic Regression model.
   - Returns a random response corresponding to the predicted intent.

## Example Interaction


User: Hello
Chatbot: Hi there!

User: Goodbye
Chatbot: Take care!


## Contribution

Feel free to fork this repository, make changes, and submit a pull request. Contributions are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [NLTK](https://www.nltk.org/) for natural language processing utilities.
- [scikit-learn](https://scikit-learn.org/) for machine learning models and vectorization.
- [Streamlit](https://streamlit.io/) for building an interactive web interface.
