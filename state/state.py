from typing import List, TypedDict, Annotated
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """Represents the state of our graph."""
    query: str
    research_data: List[str]
    development_plan: str
    code_draft: str
    final_report: str
    next: str
    messages: Annotated[List, add_messages]