# 업로드되는 파일명이 날짜/시간인 버전

import tensorflow as tf
import json
from flask import Flask, request
from flask_cors import CORS

import numpy as np
from tensorflow.keras.models import load_model
from konlpy.tag import Okt

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask('First App')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

@app.route("/analysis", methods = ['POST'])
def Analysis():
    def ScanText():
        text = request.form.get('input')
        return text
    # 분석
    model = load_model('./result_model.mod')
    okt = Okt()

    selected_words = []
    with open('selected_words.list', 'r') as file:
        selected_words = file.readlines()
    for index in range(0, len(selected_words)):
        selected_words[index] = selected_words[index].rstrip('\n')

    def tokenize(doc):
        return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

    def term_frequency(doc):
        return [doc.count(word) for word in selected_words]

    def run_review(review):
        token = tokenize(review)
        tf = term_frequency(token)
        data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
        score = float(model.predict(data))
        return str(round(score, 5))

    return run_review(ScanText())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
