import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state.state import ResearchState

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GEMINI_API_KEY)

reviewer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert reviewer. Your task is to evaluate the provided research data and determine if it is sufficient to write a comprehensive final report. Respond with only a single word: 'continue' if more research is needed, or 'end' if the data is sufficient."),
        ("user", "Here is the research data:\n\n{research_data}\n\nIs this sufficient?"),
    ]
)

reviewer_chain = reviewer_prompt | llm

def run_reviewer(state: ResearchState) -> ResearchState:
    print("Executing the Reviewer agent...")

    result = reviewer_chain.invoke({"research_data": state["research_data"]})

    content = result.content.strip().lower()

    return {**state, "next": content}