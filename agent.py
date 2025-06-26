from flask import Flask, request, jsonify
import asyncio
import json
from interview_agent import entrypoint  # Your LiveKit interview entrypoint

app = Flask(__name__)

@app.route('/')
def home():
    return "AI Interview System is running!"

# Endpoint to receive interview questions and forward to LiveKit agent
@app.route('/prewarm_agent', methods=['POST'])
async def prewarm_agent():
    # Get the questions from the request body
    data = request.get_json()
    questions = data.get('questions', [])
    
    if not questions:
        return jsonify({"status": "error", "message": "No questions provided"}), 400

    # Prewarm the agent with the questions (passing to the entrypoint)
    try:
        await entrypoint(questions)  # This will initiate the LiveKit agent with questions
        return jsonify({"status": "success", "message": "Agent prewarmed with questions"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Running the Flask application
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
