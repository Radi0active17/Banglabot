import nltk
import numpy as np
import re
from nltk.stem import PorterStemmer

# Download NLTK data if not already available
nltk.download('punkt')

# Initialize stemmer
stemmer = PorterStemmer()

def detect_language(text):
    """Detect if the text is in Bangla or English based on its characters."""
    bangla_range = (0x0980, 0x09FF)  # Unicode range for Bangla characters
    if any(bangla_range[0] <= ord(char) <= bangla_range[1] for char in text):
        return 'bengali'
    #return 'english'

def tokenize(sentence):
    """Tokenizes a sentence into words/tokens based on detected language."""
    language = detect_language(sentence)
    
    if language == 'bengali':
        # Split Bangla text based on whitespace and punctuation
        return re.findall(r'\S+', sentence)  # This regex will match all non-whitespace sequences
    else:
        # Use NLTK for English tokenization
        return nltk.word_tokenize(sentence)

def stem(word):
    """Returns the stemmed version of a word."""
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """Creates a bag of words array."""
    # Stem each word in the sentence
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    
    # Create the bag of words vector
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
