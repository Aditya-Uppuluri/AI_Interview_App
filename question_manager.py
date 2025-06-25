import json
from typing import List, Dict, Any
import google.generativeai as genai
import os

class QuestionManager:
    def __init__(self, position: str = "Software Engineer"):
        self.position = position
        self.asked_questions = []
        self.candidate_responses = []
        self.current_topic = "introduction"
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initial questions based on interview flow
        self.question_flow = {
            "introduction": [
                "Hello! Welcome to your interview. Could you please introduce yourself?",
                "Tell me about your background and experience.",
            ],
            "technical": [
                "What programming languages are you most comfortable with?",
                "Describe your experience with software development methodologies.",
            ],
            "behavioral": [
                "Tell me about a challenging project you worked on.",
                "How do you handle working under pressure?",
            ],
            "closing": [
                "Do you have any questions about the role or company?",
                "Is there anything else you'd like to share?",
            ]
        }
    
    def get_initial_question(self) -> str:
        """Get the first question to start the interview"""
        question = self.question_flow["introduction"][0]
        self.asked_questions.append(question)
        return question
    
    def generate_followup_question(self, user_response: str) -> str:
        """Generate a follow-up question based on user response"""
        self.candidate_responses.append(user_response)
        
        # Create context for Gemini
        context = self._build_context()
        
        prompt = f"""
        You are conducting a job interview for a {self.position} position.
        
        Context of the conversation so far:
        {context}
        
        The candidate just responded: "{user_response}"
        
        Based on their response, generate a relevant follow-up question that:
        1. Digs deeper into their experience
        2. Tests their technical or behavioral skills
        3. Keeps the interview flowing naturally
        4. Is appropriate for a {self.position} role
        
        Return only the question, nothing else.
        """
        
        try:
            response = self.model.generate_content(prompt)
            question = response.text.strip()
            self.asked_questions.append(question)
            return question
        except Exception as e:
            print(f"Error generating question: {e}")
            return self._get_fallback_question()
    
    def _build_context(self) -> str:
        """Build conversation context for the LLM"""
        context = []
        for i, (q, a) in enumerate(zip(self.asked_questions, self.candidate_responses)):
            context.append(f"Q{i+1}: {q}")
            context.append(f"A{i+1}: {a}")
        return "\n".join(context)
    
    def _get_fallback_question(self) -> str:
        """Get a fallback question if generation fails"""
        fallback_questions = [
            "Can you tell me more about that?",
            "What challenges did you face in that situation?",
            "How did you approach solving that problem?",
            "What would you do differently next time?",
        ]
        import random
        question = random.choice(fallback_questions)
        self.asked_questions.append(question)
        return question
    
    def should_end_interview(self) -> bool:
        """Determine if interview should end"""
        return len(self.asked_questions) >= 8  # Limit to 8 questions for demo
    
    def get_interview_summary(self) -> Dict[str, Any]:
        """Generate interview summary"""
        return {
            "position": self.position,
            "questions_asked": len(self.asked_questions),
            "questions": self.asked_questions,
            "responses": self.candidate_responses,
            "status": "completed" if self.should_end_interview() else "in_progress"
        }