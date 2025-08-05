# interface/cli.py
import os
from dotenv import load_dotenv

# --- FIX FOR MODULE NOT FOUND ERROR ---
# Add the project root directory to the Python path
import sys
# This gets the absolute path of the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# This navigates up one level to the project root
project_root = os.path.join(script_dir, '..')
# Append the project root to the system path
sys.path.append(project_root)
# -------------------------------------

from graph.main_graph import create_research_graph
from state.state import ResearchState

# Load environment variables
load_dotenv()

def run_cli():
    """
    Runs the command-line interface for the multi-agent system.
    """
    # Ensure the Google API key is set
    if not os.getenv("GEMINI_API_KEY"):  # Corrected variable name
        raise ValueError("GOOGLE_API_KEY not set. Please set it in your environment or a .env file.")

    # Create the graph instance
    app = create_research_graph()

    print("--- Multi-Agent Research and Development System ---")
    print("Enter your query below. Type 'exit' to quit.")

    while True:
        user_query = input("\nQuery: ")
        if user_query.lower() == 'exit':
            break

        # Define the initial state for the graph
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

        # Run the graph and get the final state
        final_state = app.invoke(initial_state)

        # Print the final report
        if final_state.get("final_report"):
            print("\n--- FINAL REPORT ---")
            print(final_state["final_report"])
        else:
            print("\n--- EXECUTION FAILED ---")
            print("Could not generate a final report. Check logs for details.")

if __name__ == "__main__":
    run_cli()