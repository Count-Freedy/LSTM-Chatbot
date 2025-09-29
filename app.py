from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Define the custom NotEqual layer used in your model
class NotEqual(Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs  # inputs is a list or tuple of Tensors
        return tf.not_equal(x, y)

    def get_config(self):
        base_config = super(NotEqual, self).get_config()
        return base_config

# Register custom objects for loading the model
custom_objects = {'NotEqual': NotEqual}

# Load trained model with custom_objects
model = load_model('chatbot_model.h5', custom_objects=custom_objects)

# Load tokenizers
with open('tokenizer_questions.pkl', 'rb') as f:
    tokenizer_questions = pickle.load(f)

with open('tokenizer_answers.pkl', 'rb') as f:
    tokenizer_answers = pickle.load(f)

# Load dataset for direct mapping fallback
df = pd.read_csv('data/Conversation.csv')
qa_dict = {q.lower().strip(): a for q, a in zip(df['question'], df['answer'])}

# Define max sequence lengths (adjust if used)
max_len_questions = 20
max_len_answers = 20

# Direct mapping chatbot response function (dataset dictionary)
def chatbot_response_direct_only(text):
    query = text.lower().strip()
    if not query:
        return "Please enter a message."
    return qa_dict.get(query, "Sorry, I don't know how to respond to that.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_text = request.json.get('message', '').strip()
    reply = chatbot_response_direct_only(user_text)
    return jsonify({'response': reply})

if __name__ == '__main__':
    app.run(debug=True)
