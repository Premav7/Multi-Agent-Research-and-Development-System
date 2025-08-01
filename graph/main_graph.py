from langgraph.graph import StateGraph, START, END

from state.state import ResearchState
from agents.researcher import run_researcher

# Dummy nodes (we'll replace these later)
def planning_node(state: ResearchState) -> ResearchState:
    """A dummy planning node."""
    print("Executing planning node...")
    return {**state, "development_plan": "The initial plan is to research and then write a summary."}

def reporting_node(state: ResearchState) -> ResearchState:
    """A dummy reporting node."""
    print("Executing reporting node...")
    report_content = f"Final Report based on research data: {state['research_data']}"
    return {**state, "final_report": report_content}

# The create_research_graph function
def create_research_graph():
    """Creates and compiles the LangGraph for our research system."""
    builder = StateGraph(ResearchState)

    # Add the nodes to the graph
    builder.add_node("planning", planning_node)
    builder.add_node("research", run_researcher)
    builder.add_node("reporting", reporting_node)

    # Define the flow (edges)
    builder.add_edge(START, "planning")
    builder.add_edge("planning", "research")
    builder.add_edge("research", "reporting")
    builder.add_edge("reporting", END)

    # Compile the graph
    return builder.compile()