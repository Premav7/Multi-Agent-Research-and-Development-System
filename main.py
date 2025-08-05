from dotenv import load_dotenv

from graph.main_graph import create_research_graph
from state.state import ResearchState
load_dotenv()


if __name__ == "__main__":
    initial_state = ResearchState(
        query="java code to read a file",
        research_data=[],
        development_plan="",
        code_draft="",
        final_report="",
        next="planning",
        messages=[]
    )

    app = create_research_graph()

    print("Running the graph. Check LangSmith for traces.")
    final_state = app.invoke(initial_state)

    print("\n--- Final Graph State ---")
    print(final_state)