from flask import Flask, render_template, request
from ml_logic import calculate_weighted_sentiment_score, assess_mental_state

app = Flask(__name__)

# Define your Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_quiz():
    user_responses = request.form.to_dict()  # Assuming all form fields correspond to quiz questions
    weighted_sentiment_score = calculate_weighted_sentiment_score(user_responses)
    mental_state = assess_mental_state(weighted_sentiment_score)
    return render_template('result.html', weighted_sentiment_score=weighted_sentiment_score, mental_state=mental_state)

if __name__ == '__main__':
    app.run(debug=True)
