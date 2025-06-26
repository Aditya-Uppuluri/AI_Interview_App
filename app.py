from flask import Flask, request, jsonify
from dotenv import load_dotenv
import asyncio
import threading
from livekit import agents
from livekit.plugins import google
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

async def entrypoint(questions):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=google.LLM(model="gemini-2.0-flash-exp", temperature=0.8),
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room="interviewer-bot",  # You can generate a unique room id if necessary
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )

    # Now that the agent is running, prompt it with the provided questions
    for question in questions:
        await session.generate_reply(instructions=question)

@app.route('/')
def home():
    return "AI Interview System is running!"

@app.route('/prewarm', methods=['POST'])
def prewarm():
    """API endpoint to receive questions and prewarm the model"""
    data = request.json  # Get the questions from the incoming request

    if 'questions' not in data:
        return jsonify({'error': 'No questions provided'}), 400

    questions = data['questions']  # List of questions
    
    # Run the entrypoint function in a separate thread to avoid blocking Flask's main thread
    def run_async():
        asyncio.run(entrypoint(questions))
    
    # Start the async task in a new thread
    thread = threading.Thread(target=run_async)
    thread.start()

    return jsonify({'message': 'Agent prewarm started with the provided questions'}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
