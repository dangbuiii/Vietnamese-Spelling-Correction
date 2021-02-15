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

MAXLEN = 30
NGRAM = 5
BATCH_SIZE = 64

f = open("/home/dang/Downloads/data.txt")
data = f.read()
sentences = list(sent_tokenize(data))
alphabet = '^[ _abcdefghijklmnopqrstuvwxyz0123456789áàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđ!\"\',\-\.:;?_\(\)]+$'
for s in sentences[:]:
    sentences.remove(s)
    s = s.lower()
    # s = re.sub("[0-9]+/[0-9]+", 'date', s)
    # s = re.sub("[0-9]+h[0-9]+", 'time', s)
    # s = re.sub("[0-9]+", 'number', s)
    # s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.strip()
#
#
    sentences.append(s)

def extract_pharases(text):
    return re.findall(r'\w[\w ]+', text)

phrases = itertools.chain.from_iterable(extract_pharases(text) for text in sentences)
phrases = [p.strip() for p in phrases if len(p.split()) > 1]

def autoSinhLoi(text):
    text = re.sub('lũng', 'nũng', text)
    text = re.sub('lũ', 'nũ', text)
    text = re.sub('lệch', 'nệch', text)
    text = re.sub('lẽo', 'nẽo', text)
    text = re.sub('lên', 'nên', text)
    text = re.sub('lưu', 'nưu', text)
    text = re.sub('lui', 'nui', text)
    text = re.sub('lọc', 'nọc', text)
    text = re.sub('lộng', 'nộng', text)
    text = re.sub('làng', 'nàng', text)
    text = re.sub('lay', 'nay', text)
    text = re.sub('liên', 'niên', text)
    text = re.sub('lạnh', 'nạnh', text)
    text = re.sub('lẫn', 'nẫn', text)
    text = re.sub('lũy', 'nũy', text)

    text = re.sub('nam', 'lam', text)
    text = re.sub('này', 'lày', text)
    text = re.sub('niệm', 'liệm', text)
    text = re.sub('nền', 'lền', text)
    text = re.sub('nạt', 'lạt', text)
    text = re.sub('náo', 'láo', text)
    text = re.sub('nói', 'lói', text)
    text = re.sub('nào', 'lào', text)
    text = re.sub('nồi', 'lồi', text)
    text = re.sub('nổ', 'lổ', text)
    text = re.sub('nắng', 'lắng', text)
    text = re.sub('nơi', 'lơi', text)
    text = re.sub('nới', 'lới', text)
    text = re.sub('năm', 'lăm', text)

    text = re.sub('chứng', 'trứng', text)
    text = re.sub('chợ', 'trợ', text)
    text = re.sub('chua', 'trua', text)
    text = re.sub('chị', 'trị', text)
    text = re.sub('chả', 'trả', text)
    text = re.sub('cháu', 'tráu', text)
    text = re.sub('chèn', 'trèn', text)
    text = re.sub('chỗ', 'trỗ', text)
    text = re.sub('chạm', 'trạm', text)
    text = re.sub('chiến', 'triến', text)
    text = re.sub('chắm', 'trắm', text)
    text = re.sub('chừng', 'trừng', text)
    text = re.sub('chuyền', 'truyền', text)
    text = re.sub('chuối', 'truối', text)
    text = re.sub('châm', 'trâm', text)
    text = re.sub('chưa', 'trưa', text)
    text = re.sub('chọc', 'trọc', text)

    text = re.sub('triệt', 'chiệt', text)
    text = re.sub('tru', 'chu', text)
    text = re.sub('trơn', 'chơn', text)
    text = re.sub('trắng', 'chắng', text)
    text = re.sub('trĩu', 'chĩu', text)
    text = re.sub('trệt', 'chệt', text)
    text = re.sub('trù', 'chù', text)
    text = re.sub('trụ', 'chụ', text)
    text = re.sub('tránh', 'chánh', text)
    text = re.sub('trẻ', 'chẻ', text)
    text = re.sub('trân', 'chân', text)
    text = re.sub('trật', 'chật', text)
    text = re.sub('trống', 'chống', text)
    text = re.sub('trữ', 'chữ', text)
    text = re.sub('trạng', 'chạng', text)
    text = re.sub('trú', 'chú', text)
    text = re.sub('trải', 'chải', text)
    text = re.sub('trong', 'chong', text)

    text = re.sub('sạn', 'xạn', text)
    text = re.sub('sẻ', 'xẻ', text)
    text = re.sub('sắp', 'xắp', text)
    text = re.sub('sộ', 'xộ', text)
    text = re.sub('siêu', 'xiêu', text)
    text = re.sub('sợ', 'xợ', text)
    text = re.sub('suôn', 'xuôn', text)
    text = re.sub('sướng', 'xướng', text)
    text = re.sub('siêm', 'xiêm', text)
    text = re.sub('sôi', 'xôi', text)
    text = re.sub('sam', 'xam', text)
    text = re.sub('sấm', 'xấm', text)
    text = re.sub('suông', 'xuông', text)
    text = re.sub('sắc', 'xắc', text)
    text = re.sub('sụn', 'xụn', text)
    text = re.sub('sài', 'xài', text)
    text = re.sub('sảo', 'xảo', text)
    text = re.sub('sếp', 'xếp', text)

    text = re.sub('xuyến', 'suyến', text)
    text = re.sub('xèo', 'sèo', text)
    text = re.sub('xóa', 'sóa', text)
    text = re.sub('xòe', 'sòe', text)
    text = re.sub('xóc', 'sóc', text)
    text = re.sub('xang', 'sang', text)
    text = re.sub('xứ', 'sứ', text)
    text = re.sub('xảo', 'sảo', text)
    text = re.sub('xây', 'sây', text)
    text = re.sub('xoay', 'soay', text)
    text = re.sub('xâu', 'sâu', text)
    text = re.sub('xe', 'se', text)
    text = re.sub('xong', 'song', text)
    text = re.sub('xanh', 'sanh', text)
    text = re.sub('xuốn', 'suốn', text)
    text = re.sub('xử', 'sử', text)
    text = re.sub('xã', 'sã', text)
    text = re.sub('xuông', 'suông', text)

    text = re.sub('ngã', 'ngá', text)
    text = re.sub('lũ', 'lú', text)
    text = re.sub('sĩ', 'sí', text)
    text = re.sub('lẽo', 'léo', text)
    text = re.sub('sữa', 'súa', text)
    text = re.sub('võ', 'vó', text)
    text = re.sub('bản', 'bạn', text)
    text = re.sub('đổi', 'đội', text)
    text = re.sub('não', 'náo', text)
    text = re.sub('mẩu', 'mậu', text)
    text = re.sub('bổ', 'bộ', text)
    text = re.sub('tỉ', 'tị', text)
    text = re.sub('sổm', 'sộm', text)
    text = re.sub('dở', 'dợ', text)
    text = re.sub('gửi', 'gựi', text)
    text = re.sub('nhẩm', 'nhậm', text)
    text = re.sub('rõng', 'rống', text)
    text = re.sub('phở', 'phợ', text)
    text = re.sub('tuôi', 'tui', text)
    text = re.sub('vui', 'zui', text)
    text = re.sub('vẻ', 'zẻ', text)
    text = re.sub('vời', 'zời', text)


    return text

def gen_ngrams(words,n=5):
    return ngrams(words.split(),n)
#
list_ngrams = []
for p in tqdm(phrases):
  if not re.match(alphabet, p.lower()):
    continue
  for ngr in gen_ngrams(p, NGRAM):
    if len(" ".join(ngr)) < 32:
      list_ngrams.append(" ".join(ngr))
del phrases
list_ngrams = list(set(list_ngrams))

# print(len(list_ngrams))

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
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, LSTM, Bidirectional
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

HIDDEN_SIZE = 256

model = Sequential()
model.add(LSTM(HIDDEN_SIZE, input_shape=(MAXLEN, len(alphabet)), return_sequences=True))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)))
# model.add(Flatten())
model.add(TimeDistributed(Dense(len(alphabet))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.summary()

from sklearn.model_selection import train_test_split
train_data, valid_data = train_test_split(list_ngrams, test_size=0.2,random_state=1)

def generate_data(data, batch_size=128):
    cur_index = 0
    while True:

        x, y = [], []
        for i in range(batch_size):
            y.append(encode(data[cur_index]))
            x.append(encode(autoSinhLoi(data[cur_index])))
            cur_index += 1

            if cur_index > len(data) - 1:
                cur_index = 0

        yield np.array(x), np.array(y)


train_generator = generate_data(train_data, batch_size=BATCH_SIZE)
validation_generator = generate_data(valid_data, batch_size=BATCH_SIZE)
#
checkpointer = ModelCheckpoint(filepath=os.path.join('./model3_{val_loss:.4f}_{val_accuracy:.4f}.h5_VM'),
                               save_best_only=True,
                               verbose=1)
early = EarlyStopping(patience=2, verbose=1)
#
model = model.fit_generator(train_generator, steps_per_epoch=len(train_data)//BATCH_SIZE, epochs=10,
                    validation_data=validation_generator, validation_steps=len(valid_data)//BATCH_SIZE,
                    callbacks=[checkpointer, early])