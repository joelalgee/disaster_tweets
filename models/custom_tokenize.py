import nltk
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

def tokenize(text):
    """Replace urls with placeholders, split text into lemmatized tokens.

    Args:
    text: string. The text to be tokenized.
    
    Returns:
    lemmed: list of strings. The lemmatized tokens.
    """

    # Find any urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    # Replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    
    # Normalize to lower case alphanumeric characters
    text = text.lower()
    text = re.sub("'", '', text)
    text = re.sub(r'[^a-z0-9]', ' ', text)
    
    # Tokenize
    words = word_tokenize(text)
        
    # Lemmatize nouns
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    # Lemmatize verbs
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    return lemmed