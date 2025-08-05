import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import traceable

# You can now safely remove the os.environ line.
# The `TavilySearchResults` class will automatically look for the TAVILY_API_KEY
# environment variable, which is loaded by `dotenv` in main.py.

# Create the Tavily search tool

load_dotenv()

web_search_tool = TavilySearchResults(k=5)