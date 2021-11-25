import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('punkt')


def clean_text(text: str
               ) -> list:
    pattern = re.compile(r'[^a-z]+')
    text = text.lower()
    text = pattern.sub(' ', text).strip()

    word_list = word_tokenize(text)

    stopwords_list = set(stopwords.words("english"))

    word_list = [word for word in word_list if word not in stopwords_list]
    word_list = [word for word in word_list if len(word) > 2]

    lemma = WordNetLemmatizer()
    word_list = [lemma.lemmatize(word) for word in word_list]
    return word_list
