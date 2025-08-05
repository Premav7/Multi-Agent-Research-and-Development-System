# agents/code_reviewer.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state.state import ResearchState
from langsmith import traceable

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GEMINI_API_KEY)

code_reviewer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert code reviewer. Your task is to evaluate the provided Python script and determine if it is ready for a final report. Respond with only a single word: 'correct' if the code is correct, or 'revise' if it needs revision."),
        ("user", "Here is the code to review:\n\n{code_draft}\n\nIs this code ready?"),
    ]
)

code_reviewer_chain = code_reviewer_prompt | llm

def run_code_reviewer(state: ResearchState) -> ResearchState:
    """Runs the code reviewer agent to check if the code is complete."""
    print("Executing the Code Reviewer agent...")

    result = code_reviewer_chain.invoke({"code_draft": state["code_draft"]})

    content = result.content.strip().lower()

    if content == "correct":
        return {**state, "next": "code_correct"}
    else:
        return {**state, "next": "code_revise"}