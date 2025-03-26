from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import torch
import wave
import shutil
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app) 
 # Enable CORS for frontend access

# Load Whisper model
model = whisper.load_model("tiny", device="cpu")
model.to("cpu", dtype=torch.float32)

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment_scores = analyzer.polarity_scores(text)

    arousal = sentiment_scores['compound']
    valence = polarity
    dominance = sentiment_scores['pos'] - sentiment_scores['neg']

    return {
        "arousal": arousal,
        "dominance": dominance,
        "valence": valence,
        "confidence": sentiment_scores['compound']
    }
 @app.route("/")
def home():
    return "Flask Audio Sentiment API is running!"


@app.route("/analyze-audio", methods=["POST"])
def analyze_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = f"./{file.filename}"
    
    # Save the file
    file.save(file_path)

    # Transcribe audio using Whisper
    result = model.transcribe(file_path)
    text = result["text"]

    # Perform sentiment analysis
    sentiment_result = analyze_sentiment(text)

    response = {
        "recognized_text": text,
        "sentiment_analysis": sentiment_result
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
