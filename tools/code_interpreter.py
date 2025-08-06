import subprocess
from langchain_core.tools import tool

@tool
def code_interpreter(code: str) -> str:
    """
    Executes Python code in a sandboxed environment and returns the output.
    
    Args:
        code (str): The Python code to be executed.
        
    Returns:
        str: The standard output or error from the code execution.
    """
    print("---EXECUTING CODE INTERPRETER---")
    
    try:
        process = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        return process.stdout
    except subprocess.CalledProcessError as e:
        return f"Code execution failed with error:\n{e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred during execution: {e}"