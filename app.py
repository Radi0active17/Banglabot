from flask import Flask, request, render_template
from generative_ai import generate_text  # Google Generative AI integration
from nltk_utils import tokenize, stem, bag_of_words  # NLTK helper functions
import json
import numpy as np
import random
import nltkfix

app = Flask(__name__)

# Load intents from the JSON file
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Prepare vocabulary (all words) and classes (tags)
all_words = []
tags = []
patterns = []

# Initialize conversation history
history = []

for intent in intents['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        tokenized_words = tokenize(pattern)
        all_words.extend(tokenized_words)
        patterns.append((tokenized_words, intent['tag']))

# Stem and remove duplicates
all_words = sorted(set(stem(w) for w in all_words if w.isalpha()))

def classify_intent(user_input):
    """Classifies user input into an intent based on patterns in the intents.json."""
    tokenized_input = tokenize(user_input)
    bow = bag_of_words(tokenized_input, all_words)

    # Compare input against all known patterns
    similarities = np.array([np.dot(bow, bag_of_words(tokenized_pattern, all_words)) for tokenized_pattern, _ in patterns])
    
    # Print debug information to understand the matching process
    print(f"User Input: {user_input}")
    print(f"Tokenized Input: {tokenized_input}")
    print(f"Similarities: {similarities}")

    best_match_idx = np.argmax(similarities)
    if similarities[best_match_idx] > 0:  # Ensure some words match
        matched_intent = patterns[best_match_idx][1]
        print(f"Matched Intent: {matched_intent}")
        return matched_intent
    
    return None

def get_recent_context(history, limit=6):
    """
    Get recent conversation context.
    Returns the last 'limit' messages from the history, formatted for context.
    """
    recent_history = history[-limit:]  # Get the last 'limit' messages
    context = "\n".join(recent_history)  # Join them as a string for context
    return context

@app.route('/')
def home():
    """Render the main chat interface."""
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chat():
    """Handle chat messages from the user."""
    user_input = request.form.get('msg', '').strip()

    # Store user input in history
    history.append(f"user: {user_input}")

    # Classify user input to determine intent
    intent_tag = classify_intent(user_input)

    # Use recent context for a more intelligent response
    recent_context = get_recent_context(history)

    if intent_tag:
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                response = random.choice(intent['responses'])
                
                # Add the bot's response to history
                history.append(f"bot: {response}")
                
                # Print conversation history for debugging
                print("Conversation History:", history)
                
                return response
    
    # If no intent matches, use Google Generative AI with recent context
    response = generate_text(f"Context: {recent_context}\nUser: {user_input}")
    
    # Add bot's response to the history
    history.append(f"bot: {response}")
    
    # Print conversation history for debugging
    print("Conversation History:", history)
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
