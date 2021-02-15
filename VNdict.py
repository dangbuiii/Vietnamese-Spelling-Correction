from nltk.tokenize import sent_tokenize
import string
import re
import itertools
f = open("/home/dang/Downloads/data.txt")
data = f.read()
sentences = list(sent_tokenize(data))
for s in sentences[:]:
    sentences.remove(s)
    s = s.lower()
    s = re.sub('[0-9]','',s)
    # s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.strip()
    sentences.append(s)
# print(sentences)
def extract_phrases(text):
    return re.findall(r'\w[\w ]+', text)
phrases = itertools.chain.from_iterable(extract_phrases(text) for text in sentences)
dictVN=set()
for p in phrases:
    for t in p.split():
        dictVN.add(t)
for w in dictVN:
    print(w)
