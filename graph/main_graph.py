from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import Runnable

from state.state import ResearchState
from state.state import ResearchState
from agents.researcher import run_researcher
from agents.reviewer import run_reviewer

def planning_node(state: ResearchState) -> ResearchState:
    """A dummy planning node that initializes the full state."""
    print("Executing planning node...")
    
    # We create a new, complete state object here
    # The `query` is taken from the input state, but all other keys
    # are initialized to their default values.
    return {
        "query": state["query"],
        "research_data": [],
        "development_plan": "The initial plan is to research and then write a summary.",
        "code_draft": "",
        "final_report": "",
        "next": "",
        "messages": []
    }
def reporting_node(state: ResearchState) -> ResearchState:
    print("Executing reporting node...")
    report_content = f"Final Report based on research data: {state['research_data']}"
    return {**state, "final_report": report_content}

def router(state: ResearchState):
    if state["next"] == "continue":
        return "research"
    else:
        return "reporting"

def create_research_graph() -> Runnable:
    builder = StateGraph(ResearchState)

    builder.add_node("planning", planning_node)
    builder.add_node("research", run_researcher)
    builder.add_node("review", run_reviewer)
    builder.add_node("reporting", reporting_node)

    builder.add_edge(START, "planning")
    builder.add_edge("planning", "research")
    builder.add_edge("research", "review")

    builder.add_conditional_edges(
        "review",
        router,
        {
            "research": "research",
            "reporting": "reporting"
        }
    )

    builder.add_edge("reporting", END)

    return builder.compile()