from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import os
from flask import send_from_directory
#from keras.models import load_model
#from keras.preprocessing import image
import numpy as np
#import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

app = Flask(__name__)
Bootstrap(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    df = pd.read_csv("data/datasets.csv")
    df_X = df.name
    df_Y = df.sex

    corpus = df_X
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)

    naivebayes_model = open("models/naivebayesgendermodel.pkl", "rb")
    clf = joblib.load(naivebayes_model)


    if request.method == 'POST':
        namequery = request.form['namequery']
        print(namequery)
        data = [namequery]
        vect = cv.transform(data).toarray()
        myprediction = clf.predict(vect)
        print(myprediction)
    return render_template("result.html", prediction=myprediction, name=namequery.upper())



if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
