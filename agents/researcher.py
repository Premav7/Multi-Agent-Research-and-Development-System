# agents/researcher.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor

# We are importing the corrected tool
from tools.web_search import web_search_tool
from state.state import ResearchState
from langsmith import traceable

# Retrieve the API key from the environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up the LLM, explicitly passing the API key

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0,
    google_api_key=GEMINI_API_KEY # <-- This is the fix for the credentials error
)

# Define the tools available to this agent
tools = [web_search_tool]

# Create the prompt template for the agent
research_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world-class researcher. Your goal is to gather information about the user's query and provide a comprehensive answer."),
        ("placeholder", "{messages}"),
        ("user", "Conduct research on: {query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the agent
researcher_agent = create_tool_calling_agent(llm, tools, research_prompt)

# Define a function to run the agent
@traceable(name="researcher")
def run_researcher(state: ResearchState) -> ResearchState:
    """Runs the researcher agent to gather data."""
    print("Executing the Researcher agent (Gemini)...")
    
    agent_executor = AgentExecutor(agent=researcher_agent, tools=tools, verbose=True)
    
    # Run the agent with the user's query
    result = agent_executor.invoke({"messages": [], "query": state["query"]})

    # Update the state with the research result.
    new_message = result["output"]
    
    return {**state, "research_data": state["research_data"] + [new_message]}