from flask import Flask,render_template,request
import re
import Model_LoiVungMien
import Model_LoiKhongDau
import Model_LoiDanhMay
from nltk.tokenize import sent_tokenize


app = Flask(__name__)

@app.route("/",methods =['GET'])
def test():
    return render_template('index.html')
@app.route("/action1",methods=['POST'])
def action1():
    keyword = request.form['keyword']
    sentence = sent_tokenize(keyword)
    l=[]
    for s in sentence:
        s = re.sub(";", "", s)
        s = re.sub("['(',')']", "", s)
        s =s.strip()
        kw = Model_LoiVungMien.attach_font(s)
        l.append(kw)
    result=' '.join(l)
    return result
@app.route("/action2",methods=['POST'])
def action2():
    keyword = request.form['keyword']
    sentence = sent_tokenize(keyword)
    l = []
    for s in sentence:
        s = re.sub(";", "", s)
        s = re.sub("['(',')']", "", s)
        s = s.strip()
        kw = Model_LoiKhongDau.attach_font(s)
        l.append(kw)
    result = ' '.join(l)

    return result
@app.route("/action3",methods=['POST'])
def action3():
    keyword = request.form['keyword']
    sentence = sent_tokenize(keyword)
    l = []
    for s in sentence:
        s = re.sub(";", "", s)
        s = re.sub("['(',')']", "", s)
        s = s.strip()
        kw = Model_LoiDanhMay.attach_font(s)
        l.append(kw)
    result = ' '.join(l)

    return result
@app.route("/action4",methods=['POST'])
def action4():
    keyword = request.form['keyword']
    check = sent_tokenize(keyword)
    sentence = sent_tokenize(keyword)
    l = []
    for s in sentence:
        s = re.sub(";", "", s)
        s = re.sub("['(',')']", "", s)
        s = s.strip()
        s = Model_LoiDanhMay.accent_sentence(s)
        s = Model_LoiVungMien.accent_sentence(s)
        l.append(s)
    for i in range(0,len(check)):
        l[i] = Model_LoiVungMien.forAc4(l[i],check[i])
    result = ' '.join(l)

    return result

if __name__ == "__main__":
    app.run()