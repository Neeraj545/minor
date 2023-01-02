import string
import nltk
from flask import Flask,request,jsonify
import pickle
import sklearn

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

model = pickle.load(open('model.pkl','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    # converting into lower case
    text = text.lower()

    # converting the text into seperate words
    text = nltk.word_tokenize(text)

    # removing special characters
    temp = []

    for word in text:
        if word.isalnum():
            temp.append(word)

    text = temp[:]
    temp.clear()

    # removing all the stopwords and removing punctuations
    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            temp.append(word)

    text = temp[:]
    temp.clear()

    # performing stemming in the text
    porterstemmer = PorterStemmer()
    for word in text:
        temp.append(porterstemmer.stem(word))
    text = temp[:]

    return " ".join(text)

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict',methods=['POST'])
def predict():
    message = request.form.get('msg')

    transformed_sms = transform_text(message)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    classification_result = {"classification":str(result)}
    return jsonify(classification_result)


if __name__ == '__main__':
    app.run(debug=True)

