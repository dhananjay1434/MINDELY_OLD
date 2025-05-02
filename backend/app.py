import os
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any

import bcrypt
import google.generativeai as genai
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "False").lower() == "true" else logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())

# Configure CORS
cors_origin = os.getenv("CORS_ALLOW_ORIGIN", "*")
CORS(app, resources={r"/api/*": {"origins": cors_origin}})

# Configure API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=GEMINI_API_KEY)

# Log application startup
logging.info(f"Starting AI Friend Backend API with CORS allowed for: {cors_origin}")

# Global variables
_SENTENCE_TRANSFORMER_LOCK = threading.RLock()
_SENTENCE_TRANSFORMER_INSTANCE = None
_EMBEDDING_DIMENSION = None

# In-memory user storage (replace with database in production)
users = {}
chat_history = defaultdict(list)

# Rate limiter
class RateLimiter:
    def __init__(self, rate=10, per=60, burst=20):
        self.rate = rate
        self.per = per
        self.burst = burst
        self.buckets = defaultdict(lambda: {"tokens": burst, "last_refill": time.time()})
        self.lock = threading.RLock()

    def check_rate_limit(self, identifier):
        with self.lock:
            bucket = self.buckets[identifier]
            now = time.time()

            time_passed = now - bucket["last_refill"]
            new_tokens = time_passed * (self.rate / self.per)
            bucket["tokens"] = min(self.burst, bucket["tokens"] + new_tokens)
            bucket["last_refill"] = now

            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True, 0
            else:
                retry_after = (1 - bucket["tokens"]) * (self.per / self.rate)
                return False, retry_after

# Initialize rate limiter
rate_limiter = RateLimiter(rate=5, per=60, burst=10)

# Embedding functions
def get_sentence_transformer(model_name='all-MiniLM-L6-v2'):
    global _SENTENCE_TRANSFORMER_INSTANCE, _EMBEDDING_DIMENSION

    with _SENTENCE_TRANSFORMER_LOCK:
        if _SENTENCE_TRANSFORMER_INSTANCE is None:
            logging.info(f"Loading Sentence Transformer model: {model_name}")

            try:
                _SENTENCE_TRANSFORMER_INSTANCE = SentenceTransformer(model_name)

                if torch.cuda.is_available():
                    _SENTENCE_TRANSFORMER_INSTANCE = _SENTENCE_TRANSFORMER_INSTANCE.to(torch.device("cuda"))
                    logging.info("Using CUDA for sentence transformer acceleration")

                _EMBEDDING_DIMENSION = _SENTENCE_TRANSFORMER_INSTANCE.get_sentence_embedding_dimension()
                logging.info(f"Model loaded successfully (Dim: {_EMBEDDING_DIMENSION})")

            except Exception as e:
                logging.error(f"Failed to load model: {e}")
                raise

    return _SENTENCE_TRANSFORMER_INSTANCE

@lru_cache(maxsize=1024)
def get_embedding(text, model=None):
    if model is None:
        model = get_sentence_transformer()

    if not isinstance(text, str) or not text.strip():
        return None

    try:
        with torch.no_grad():
            embedding = model.encode(
                text.strip(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

        return embedding.astype(np.float32)

    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

# Authentication functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(stored_hash, password):
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))

# Chat functions
def generate_response(user_id, message):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Get chat history
        history = chat_history[user_id][-5:]  # Last 5 messages for context

        # Format history for the model
        formatted_history = []
        for msg in history:
            formatted_history.append({"role": msg["role"], "parts": [msg["content"]]})

        # Add the new message
        formatted_history.append({"role": "user", "parts": [message]})

        # Generate response
        chat = model.start_chat(history=formatted_history)
        response = chat.send_message(message)

        # Save to history
        chat_history[user_id].append({"role": "user", "content": message, "timestamp": datetime.now(timezone.utc).isoformat()})
        chat_history[user_id].append({"role": "assistant", "content": response.text, "timestamp": datetime.now(timezone.utc).isoformat()})

        return response.text

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm sorry, I encountered an error processing your request. Please try again."

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required"}), 400

    if username in users:
        return jsonify({"success": False, "message": "Username already exists"}), 400

    user_id = str(uuid.uuid4())
    users[username] = {
        "user_id": user_id,
        "password_hash": hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    return jsonify({"success": True, "user_id": user_id, "message": "Registration successful"})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required"}), 400

    if username not in users:
        return jsonify({"success": False, "message": "Invalid username or password"}), 401

    if not verify_password(users[username]["password_hash"], password):
        return jsonify({"success": False, "message": "Invalid username or password"}), 401

    return jsonify({
        "success": True,
        "user_id": users[username]["user_id"],
        "message": "Login successful"
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id')
    message = data.get('message')

    # Check if user exists
    user_exists = False
    for user_data in users.values():
        if user_data["user_id"] == user_id:
            user_exists = True
            break

    if not user_exists:
        return jsonify({"success": False, "message": "Invalid user ID"}), 401

    # Apply rate limiting
    allowed, retry_after = rate_limiter.check_rate_limit(user_id)
    if not allowed:
        return jsonify({
            "success": False,
            "message": f"Rate limit exceeded. Please try again in {int(retry_after)} seconds."
        }), 429

    # Generate response
    response = generate_response(user_id, message)

    return jsonify({
        "success": True,
        "response": response,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    user_id = request.args.get('user_id')

    # Check if user exists
    user_exists = False
    for user_data in users.values():
        if user_data["user_id"] == user_id:
            user_exists = True
            break

    if not user_exists:
        return jsonify({"success": False, "message": "Invalid user ID"}), 401

    return jsonify({
        "success": True,
        "history": chat_history[user_id]
    })

if __name__ == '__main__':
    # Load model on startup
    get_sentence_transformer()

    # Get port from environment variable
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'

    # Log startup information
    logging.info(f"Starting server on port {port} with debug={debug_mode}")

    # Run the app
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
