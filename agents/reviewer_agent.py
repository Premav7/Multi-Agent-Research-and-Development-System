import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state.state import ResearchState

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set. Please set it in your environment or a .env file.")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GEMINI_API_KEY)

review_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an internal reviewer for a multi-agent system. Your task is to review the research data provided and decide if it is sufficient and relevant to answer the user's query.

    Respond with a single word:
    - 'PROCEED' if the research data is good enough to move forward.
    - 'REVISE' if the research data is incomplete, irrelevant, or requires more detail.
    
    Research Data: {research_data}
    User Query: {query}
    """),
    ("user", "Decision:")
])
review_chain = review_prompt | llm

# agents/reviewer_agent.py
# ... (existing imports)

def run_reviewer_agent(state: ResearchState) -> ResearchState:
    """
    Runs the reviewer agent to check if the research data is sufficient.
    """
    print("--- REVIEWER AGENT: Reviewing research data ---")

    research_data = state["research_data"]
    query = state["query"]

    try:
        response = review_chain.invoke({"research_data": research_data, "query": query})
        decision = response.content.strip().upper()
    except Exception as e:
        print(f"An error occurred while invoking the LLM: {e}")
        decision = "REVISE" 

    if "PROCEED" in decision:
        print("Reviewer Agent decided to proceed.")
        state["internal_review_decision"] = "proceed"
    else:
        print("Reviewer Agent decided to revise.")
        state["internal_review_decision"] = "revise"

    return state