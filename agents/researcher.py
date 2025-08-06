import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools.web_search import web_search_tool
from state.state import ResearchState
from langsmith import traceable

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set. Please set it in your environment or a .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0,
    google_api_key=GEMINI_API_KEY 
)

tools = [web_search_tool]

research_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world-class researcher. Your goal is to gather information about the user's query and provide a comprehensive answer."),
        ("placeholder", "{messages}"),
        ("user", "Conduct research on: {query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

researcher_agent = create_tool_calling_agent(llm, tools, research_prompt)

@traceable(name="researcher")
def run_researcher(state: ResearchState) -> ResearchState:
    """Runs the researcher agent to gather data."""
    print("Executing the Researcher agent (Gemini)...")
    
    try:
        agent_executor = AgentExecutor(agent=researcher_agent, tools=tools, verbose=True)
        result = agent_executor.invoke({"messages": [], "query": state["query"]})
        new_message = result["output"]
    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
        return {**state, "research_data": state["research_data"] + [f"Error: {e}"]}

    return {**state, "research_data": state["research_data"] + [new_message]}