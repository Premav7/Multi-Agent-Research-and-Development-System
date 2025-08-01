from dotenv import load_dotenv
# It's crucial that this line is at the very top, before any other imports
# that might access environment variables.
load_dotenv()

from graph.main_graph import create_research_graph
from state.state import ResearchState

if __name__ == "__main__":
    # Create an initial state for the graph
    initial_state = ResearchState(
        query="the viability of decentralized energy grids",
        research_data=[],
        development_plan="",
        code_draft="",
        final_report="",
        next="planning",
        messages=[]
    )

    # Build and compile the graph
    app = create_research_graph()

    # Run the graph with the initial state
    print("Running the graph. Check LangSmith for traces.")
    final_state = app.invoke(initial_state)

    print("\n--- Final Graph State ---")
    print(final_state)