import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from state.state import ResearchState
from langsmith import traceable

# Load the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set. Please set it in your environment or a .env file.")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=GEMINI_API_KEY)

reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional technical report writer. Your task is to compile a final report based on provided research data and generated code. Ensure the report is well-structured, easy to read, and provides a clear summary of the findings and the code solution.
    
    The report should have the following sections:
    1.  **Summary:** A brief overview of the solution.
    2.  **Core Concepts:** Key ideas and technologies used.
    3.  **Code Examples:** The full, commented code snippets for the solution.
    4.  **Important Considerations:** A list of best practices and warnings related to the code.
    
    Format the output using markdown for clear headings and code blocks.
    """),
    ("user", """
    Research Data: {research_data}
    
    Generated Code:
    {code_draft}
    """)
])

reviewer_chain = reviewer_prompt | llm


@traceable(name="reviewer")
def run_reviewer(state: ResearchState) -> ResearchState:
    """
    Runs the reviewer agent to generate the final report from research and code.
    """
    print("---REVIEWER AGENT: Generating final report---")
    
    research_data_str = "\n".join(state.get("research_data", []))
    code_content = state.get("code_draft") or "No code was generated for this query."
    
    input_data = {
        "research_data": research_data_str,
        "code_draft": code_content
    }
    
    try:
        final_report = reviewer_chain.invoke(input_data)
        report_content = final_report.content
        return {**state, "final_report": report_content, "next": "end"}
    except Exception as e:
        print(f"Error invoking reviewer chain: {e}")
        return {**state, "final_report": "Error generating final report.", "next": "end"}