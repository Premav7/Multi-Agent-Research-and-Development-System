# agents/coder.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from state.state import ResearchState
from langsmith import traceable

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=GEMINI_API_KEY)

coder_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert Python programmer. Your task is to write a Python script that addresses the user's query, using the provided research data. The output should be only the executable code itself, without any introductory or concluding text, or markdown code fences."),
        ("user", "Here is the research data:\n\n{research_data}\n\nWrite a Python script to address the user's query: {query}"),
    ]
)

coder_chain = coder_prompt | llm

@traceable(name="coder")
def run_coder(state: ResearchState) -> ResearchState:
    """Runs the coder agent to generate a Python script."""
    print("Executing the Coder agent...")

    result = coder_chain.invoke({"research_data": state["research_data"], "query": state["query"]})

    code_draft = result.content.strip()

    # The code draft is saved in the state
    return {**state, "code_draft": code_draft}