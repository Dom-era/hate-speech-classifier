from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('hate_speech_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)

    return f'This message is categorized as: {prediction[0]}'

if __name__ == '__main__':
    app.run(debug=True)
