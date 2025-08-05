import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import traceable


load_dotenv()

web_search_tool = TavilySearchResults(k=5)