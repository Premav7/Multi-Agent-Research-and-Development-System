
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import Runnable
from langsmith import traceable

import os

from state.state import ResearchState
from agents.researcher import run_researcher
from agents.reviewer import run_reviewer
from agents.coder import run_coder
from agents.coder_reviewer import run_code_reviewer
from agents.reviewer_agent import run_reviewer_agent 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GEMINI_API_KEY)


def planning_node(state: ResearchState) -> ResearchState:
    """A dummy planning node that initializes the full state."""
    print("Executing planning node...")
    return {
        "query": state["query"],
        "research_data": [],
        "development_plan": "The initial plan is to research and then write a summary.",
        "code_draft": "",
        "final_report": "",
        "next": "",
        "messages": []
    }

def internal_review_router(state: ResearchState):
    """
    Routes the graph based on the internal reviewer agent's decision.
    """
    if state["internal_review_decision"] == "proceed":
        print("Internal review passed, proceeding to research router.")
        return "continue"
    else:
        print("Internal review failed, re-running research agent.")
        return "revise_research"

def reporting_node(state: ResearchState) -> ResearchState:
    print("Executing reporting node...")
    return run_reviewer(state)


router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a router for a multi-agent system. Your task is to analyze a user's query and decide whether it requires code generation or just a final report.
    
    Respond with a single word: 'CODE' if the query is a request for a script, program, function, or any form of code.
    Respond with a single word: 'REPORT' if the query is a general question or a request for information that does not require code.
    
    Examples:
    - "write a Python script to find prime numbers" -> CODE
    - "tell me about the history of space travel" -> REPORT
    - "what is the best way to write a C++ program for a linked list" -> CODE
    - "summarize the latest findings about black holes" -> REPORT
    - "I need a solution for a factorial function" -> CODE
    """),
    ("user", "{query}")
])
router_chain = router_prompt | llm

@traceable(name="research_router")
def research_router(state: ResearchState):
    """
    This router uses an LLM to determine if the graph should proceed
    to code generation or reporting based on the user's query.
    """
    print("---LLM ROUTER: Analyzing query intent---")
    
    query = state["query"]
    
    response = router_chain.invoke({"query": query})
    
    if "CODE" in response.content.upper():
        print("Query requires code. Routing to code_generation.")
        return "code_generation"
    else:
        print("Query does not require code. Routing to reporting.")
        return "reporting"

def code_router(state: ResearchState):
    if state["next"] == "code_revise":
        return "code_generation"
    else:
        return "reporting"
    
def create_research_graph() -> Runnable:
    builder = StateGraph(ResearchState)
    builder.add_node("planning", planning_node)
    builder.add_node("research", run_researcher)
    builder.add_node("internal_review", run_reviewer_agent) 
    builder.add_node("code_generation", run_coder)
    builder.add_node("review_code", run_code_reviewer)
    builder.add_node("reporting", reporting_node)
    
    def continue_node(state: ResearchState) -> ResearchState:
        return state
    builder.add_node("continue_to_router", continue_node)

    builder.add_edge(START, "planning")
    builder.add_edge("planning", "research")
    builder.add_edge("research", "internal_review")

    builder.add_conditional_edges(
        "internal_review",
        internal_review_router,
        {
            "continue": "continue_to_router", 
            "revise_research": "research",   
        }
    )

    builder.add_conditional_edges(
        "continue_to_router",
        research_router,
        {
            "code_generation": "code_generation",
            "reporting": "reporting"
        }
    )

    builder.add_conditional_edges(
        "review_code",
        code_router,
        {
            "code_generation": "code_generation",
            "reporting": "reporting"
        }
    )

    builder.add_edge("code_generation", "review_code")
    builder.add_edge("reporting", END)

    return builder.compile()