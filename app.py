from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

file_name = 'Spam_classifier.pkl'
classifier = pickle.load(open(file_name, 'rb'))
tf = TfidfVectorizer(max_features=500)
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        temp = tf.transform(data).toarray()
        my_predict = classifier.predict(temp)
        return render_template('result.html', prediction=my_predict)


if __name__ == '__main__':
    app.run(debug=True)
