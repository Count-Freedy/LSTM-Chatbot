# LSTM Chatbot

## Description
This project is an LSTM-based chatbot built using a sequence-to-sequence neural network model trained on a custom dataset of conversational pairs. The chatbot answers general questions and replies based on the trained dataset.
(LSTM - Long Short-term Memory)

## Features
  - Encoder-Decoder LSTM model for conversational AI
  - Dataset-driven direct mapping for quick, accurate responses
  - Flask backend serving the chatbot model as a web API
  - Simple web-based user interface for chatting with the bot
  - Code includes data preprocessing, model training, and inference

## Dataset
The dataset (Conversation.csv) contains question-answer pairs used to train the chatbot. Each answer is framed with start and end tokens <start> and <end> to aid training.

## Getting Started
1. Prerequisites:
  - Python 3.8+
  - TensorFlow
  - Flask
  - Pandas

2. Installation
   
1. Clone the repository

2. Create and activate a virtual environment (recommended):
     ```bash
     python -m venv chatbot_env
     source chatbot_env/bin/activate  # Linux/Mac
     chatbot_env\Scripts\activate     # Windows

3. Install dependencies:
    ```bash
    pip install -r requirements.txt

4. Start the Flask app:
   ```bash
   python app.py

## Training
The project includes a script to train the chatbot model on the provided dataset using an encoder-decoder LSTM architecture. Trained model and tokenizers are saved for inference.

## Notes
  - Model saving/loading uses a custom layer NotEqual. Make sure this layer code is present during loading.
  - The current deployment uses direct question-answer mapping for response generation.
  - Further improvements can include better model inference and deployment scaling.

