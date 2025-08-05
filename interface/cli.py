import os
from dotenv import load_dotenv

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.append(project_root)

from graph.main_graph import create_research_graph
from state.state import ResearchState

load_dotenv()

def run_cli():
    """
    Runs the command-line interface for the multi-agent system.
    """
    if not os.getenv("GEMINI_API_KEY"): 
        raise ValueError("GOOGLE_API_KEY not set. Please set it in your environment or a .env file.")

    app = create_research_graph()

    print("--- Multi-Agent Research and Development System ---")
    print("Enter your query below. Type 'exit' to quit.")

    while True:
        user_query = input("\nQuery: ")
        if user_query.lower() == 'exit':
            break

        initial_state = ResearchState(
            query=user_query,
            research_data=[],
            development_plan="",
            code_draft="",
            final_report="",
            next="",
            messages=[]
        )

        print("\nStarting analysis. This may take a moment...\n")

        final_state = app.invoke(initial_state)

        if final_state.get("final_report"):
            print("\n--- FINAL REPORT ---")
            print(final_state["final_report"])
        else:
            print("\n--- EXECUTION FAILED ---")
            print("Could not generate a final report. Check logs for details.")

if __name__ == "__main__":
    run_cli()