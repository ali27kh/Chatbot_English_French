import os
import random
import json
import pickle
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="Chatbot API", description="Chatbot powered by FastAPI", version="1.0")

# Paths
lemmatizer = WordNetLemmatizer()
intents_path = "mbi_questions_english.json"
words_path = "words.pkl"
classes_path = "classes.pkl"
model_path = "chatbot_model_eng.keras"

# Verify files exist
for path, desc in [(intents_path, "Intents file"), 
                   (words_path, "Words file"), 
                   (classes_path, "Classes file"), 
                   (model_path, "Model file")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} not found: {path}")

# Load resources
with open(intents_path, "r", encoding="utf-8") as f:
    intents = json.load(f)

words = pickle.load(open(words_path, "rb"))
classes = pickle.load(open(classes_path, "rb"))

# Load full model (architecture + weights)
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Model loaded successfully from '{model_path}'")
except (ValueError, OSError) as e:
    raise RuntimeError(f"Failed to load the model. Ensure '{model_path}' is a full saved model.") from e

# Pydantic model for request
class Message(BaseModel):
    text: str

# Utilities
def clean_up_sentence(sentence: str):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens]

def bag_of_words(sentence: str):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if w in sentence_words else 0 for w in words])

def predict_class(sentence: str):
    bow = bag_of_words(sentence)
    print(f"Bag of words: {bow}")  # Débogage
    res = model.predict(np.array([bow]), verbose=0)[0]
    print(f"Model output: {res}")  # Débogage
    ERROR_THRESHOLD = 0.1  # Réduit pour capturer plus d'intentions
    results = [{"intent": classes[i], "probability": float(r)} 
               for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    print(f"Filtered intents: {results}")  # Débogage
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I don't understand your question, please try again."
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I don't understand your question, please try again."

# Routes
@app.get("/")
def root():
    return {"message": "Chatbot API is running!"}

@app.get("/chat")
def chat_get(text: str):
    try:
        print(f"Received message (GET): {text}")  # Débogage
        ints = predict_class(text)
        print(f"Predicted intents: {ints}")  # Débogage
        res = get_response(ints, intents)
        print(f"Response: {res}")  # Débogage
        return {"user_message": text, "response": res, "intent": ints}
    except Exception as e:
        print(f"Error: {str(e)}")  # Débogage
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/chat")
def chat_get(text: str):
    try:
        print(f"Received message (GET): {text}")  # Débogage
        ints = predict_class(text)
        print(f"Predicted intents: {ints}")  # Débogage
        res = get_response(ints, intents)
        print(f"Response: {res}")  # Débogage
        return {"user_message": text, "response": res, "intent": ints}
    except Exception as e:
        print(f"Error: {str(e)}")  # Débogage
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")