from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
tfidf = TfidfVectorizer(stop_words='english',max_df=0.7)
loaded_model = pickle.load(open('model3.pkl','rb'))
df = pd.read_csv('news.csv')
x = df['text']
y = df['label']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

def fake_news_det(news):
    tfid_x_train = tfidf.fit_transform(x_train)
    tfidf_x_test =  tfidf.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfidf.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html',prediction=pred)
    else:
        return render_template('index.html',prediction="Something went wrong")

if __name__ == '__main__':
        app.run(debug=True)