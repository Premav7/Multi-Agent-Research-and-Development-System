from fastapi import APIRouter
from model.fastapi_model import ModelInput
from graph.main_graph import create_research_graph
from state.state import ResearchState

router = APIRouter()
app = create_research_graph()

@router.post("/invoke")
async def invoke_graph(input_data: ModelInput):

    initial_state = ResearchState(
        query=input_data.query,
        research_data=[],
        development_plan="",
        code_draft="",
        final_report="",
        next="",
        messages=[]
    )

    final_state = app.invoke(initial_state)

    if final_state.get("final_report"):
        return {"final_report": final_state["final_report"]}
    else:
        return {"error": "Could not generate a final report. Check logs for details."}