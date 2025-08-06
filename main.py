from dotenv import load_dotenv
from fastapi import FastAPI
from interface.fastapi_interface import router  

from graph.main_graph import create_research_graph
from state.state import ResearchState
load_dotenv()

app = FastAPI(
    title="Multi-Agent Research System API",
    description="An API to run a multi-agent research and development graph.",
    version="1.0.0",
)

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-Agent Research System API!"}

