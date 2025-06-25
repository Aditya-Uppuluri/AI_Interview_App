import asyncio
import json
import os
from typing import Optional
from livekit import agents, rtc
from livekit.agents import (
    Agent, 
    AgentSession, 
    ChatContext, 
    ChatMessage,
    RunContext,
    function_tool
)
from livekit.plugins import google  # Correct import for Google Gemini
# from livekit.agents.llm import ChatLLM  # If needed for fallback
import google.generativeai as genai
from question_manager import QuestionManager



class InterviewAgent(Agent):
    def __init__(self, chat_ctx: ChatContext) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions="""You are an AI interview assistant conducting a professional job interview. 
            Be professional, friendly, and engaging. Ask follow-up questions based on responses. 
            Keep questions relevant to the job position. Listen carefully and provide natural responses."""
        )
        self.question_manager = QuestionManager(
            position=os.getenv("INTERVIEW_POSITION", "Software Engineer")
        )
        self.interview_active = True
    
    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Called when user completes their turn"""
        if not self.interview_active:
            return
            
        user_response = new_message.text_content()
        print(f"User response: {user_response}")
        
        # Check if interview should end
        if self.question_manager.should_end_interview():
            await self._end_interview(turn_ctx)
            return
        
        # Generate next question based on response
        next_question = self.question_manager.generate_followup_question(user_response)
        
        # Add context for the assistant's next response
        turn_ctx.add_message(
            role="system",
            content=f"Ask this question naturally: {next_question}"
        )
    
    async def _end_interview(self, ctx: ChatContext):
        """End the interview gracefully"""
        self.interview_active = False
        summary = self.question_manager.get_interview_summary()
        
        ctx.add_message(
            role="system",
            content="End the interview by thanking the candidate and letting them know next steps will be communicated soon."
        )
        
        print("Interview Summary:", json.dumps(summary, indent=2))
    
    @function_tool()
    async def get_next_question(self, context: RunContext) -> str:
        """Tool to get the next interview question"""
        if not hasattr(self, '_first_question_asked'):
            question = self.question_manager.get_initial_question()
            self._first_question_asked = True
            return question
        return "Continue with the natural flow of conversation."

# Entry point to run the agent
async def entrypoint(ctx: agents.JobContext):
    """Main entry point for the interview agent"""
    
    # Get job metadata if available
    metadata = {}
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
        except json.JSONDecodeError:
            pass
    
    candidate_name = metadata.get("candidate_name", "Candidate")
    position = metadata.get("position", os.getenv("INTERVIEW_POSITION", "Software Engineer"))
    
    print(f"Starting interview for {candidate_name} for position: {position}")
    
    await ctx.connect()
    
    # Initialize components (use Google Gemini)
    llm = google.LLM(
        model="gemini-2.0-flash-exp",  # Choose the model you need
        temperature=0.7,  # You can tune this as per requirement
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create initial context
    initial_ctx = ChatContext()
    initial_ctx.add_message(
        role="system", 
        content=f"You are interviewing {candidate_name} for a {position} position. Start with a warm greeting."
    )
    
    # Create and start session with Google Gemini LLM
    session = AgentSession(
        llm=llm,
        vad=rtc.VAD.SPEECH_BRAIN,
        turn_detection=agents.TurnDetection(
            use_speech_interruption=True,
            min_speech_length=1.0,
            max_silence_length=2.0
        )
    )
    
    agent = InterviewAgent(chat_ctx=initial_ctx)
    
    await session.start(
        room=ctx.room,
        agent=agent
    )
    
    # Start the interview with initial greeting
    await session.generate_reply(
        instructions=f"""Greet {candidate_name} warmly and ask the first interview question: 
        '{agent.question_manager.get_initial_question()}'"""
    )
    
    # Keep the session running
    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    # For local testing
    agents.cli.run_app(entrypoint)
