import re
from nltk.tokenize import sent_tokenize
from nltk import ngrams
from collections import Counter
import string
import random
import numpy as np
import os
import itertools
from tqdm import tqdm
# import unidecode
from operator import itemgetter



MAXLEN = 18
NGRAM = 3
BATCH_SIZE = 32


def extract_pharases(text):
    return re.findall(r'\w[\w ]+', text)


def gen_ngrams(words,n=5):
    return ngrams(words.split(),n)


accented_chars_vietnamese = [
    'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
    'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
    'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
    'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
    'í', 'ì', 'ỉ', 'ĩ', 'ị',
    'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
    'đ',
]

accented_chars_vietnamese.extend([c.upper() for c in accented_chars_vietnamese])
alphabet = list(('\x00 _' + string.ascii_letters + string.digits + ''.join(accented_chars_vietnamese)))
#
def encode(text,maxlen= MAXLEN):
    text = "\x00" +text
    x = np.zeros((maxlen, len(alphabet)))
    for i, c in enumerate(text[:maxlen]):
        x[i, alphabet.index(c)] = 1
    if i < maxlen - 1:
        for j in range(i + 1, maxlen):
            x[j, 0] = 1
    return x
def decode(x, calc_argmax=True):
    if calc_argmax:
        x = x.argmax(axis=-1)
    return ''.join(alphabet[i] for i in x)
#
from tensorflow.keras.models import load_model
model = load_model("./model_0.0094_0.9974.h5_64_1")


def extract_phrases(text):
    pattern = r'\w[\w ]*|\s\W+|\W+'
    return re.findall(pattern, text)

def guess(ngram):
    text = ' '.join(ngram)
    preds = model.predict(np.array([encode(text)]), verbose=0)
    return decode(preds[0], calc_argmax=True).strip('\x00')


def add_accent(text):
    ngrams = list(gen_ngrams(text.lower(), n=NGRAM))
    guessed_ngrams = list(guess(ngram) for ngram in ngrams)
    candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(' +', ngram)):
            candidates[nid + wid].update([word])
    output = ' '.join(c.most_common(1)[0][0] for c in candidates)
    return output

def accent_sentence(sentence):
    list_phrases = extract_phrases(sentence)
    output = ""
    for phrases in list_phrases:
        if len(phrases.split()) < 2 or not re.match("\w[\w ]+", phrases):
            output += phrases
        else:
            output += add_accent(phrases)
        if phrases[-1] == " ":
            output += " "
    return output

def attach_font(keyword):
    sentence = accent_sentence(keyword)
    tokens = sentence.split()
    wrong = keyword.split()
    resultHtml = "<p>"
    for i in range(0, len(tokens)):
        if (tokens[i] == wrong[i]):
            resultHtml = resultHtml +  tokens[i] +" "
        else:
            resultHtml = resultHtml + "<span style='color:red;font-weight:bold'>" + tokens[i] + "</span>"+" "

    resultHtml = resultHtml + "</p>"
    return resultHtml







