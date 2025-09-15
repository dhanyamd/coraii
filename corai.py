import os
import re 
import json
import textwrap 
import base64 
from pathlib import Path 
from typing import List, Dict, Any, Optional, Union 
from together import Together

reasoning_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
max_iterations = 15  # Maximum number of reasoning cycles
temperature = 0.2    # Lower temperature for more consistent code generation
session_timeout = 3600  # Session timeout in seconds

# Display Settings
max_output_length = 500  # Maximum words in displayed output
show_images = True      # Whether to display generated plots
box_width = 80         # Width of output boxes

together_client = Together(api_key="--your--together-api-key--")  # Replace with your API key
code_interpreter = together_client.code_interpreter 

def run_python(code: str, session_id: Optional[str] = None, files: Optional[list[Dict[str, str]]] = None):
    """
    Executes Python code using Together Code Interpreter and returns the result.
    Args:
        code: The Python code to execute
        session_id: Optional session ID to maintain state between executions
        files: Optional list of files to upload to the code interpreter
              Each file should be a dict with 'name', 'encoding', and 'content' keys

    Returns:
        The output of the executed code as a JSON
    """
    try:
        kwargs = {"code": code, "language": "python"}

        if session_id:
            kwargs["session_id"] = session_id

        if files:
            kwargs["files"] = files

        response = code_interpreter.run(**kwargs)

        result = {"session_id": response.data.session_id, "status": response.data.status, "outputs": []}

        for output in response.data.outputs:
            result["outputs"].append({"type": output.type, "data": output.data})

        if response.data.errors:
            result["errors"] = response.data.errors

        return result
    except Exception as e:
        error_result = {"status": "error", "error_message": str(e), "session_id": None}
        return error_result
    
def collect_files(directory) -> list[Dict[str, str]]:
    """
    Collects all files from the specified directory and its subdirectories.

    Args:
        directory: The directory to scan for files

    Returns:
        A list of file dictionaries ready for upload to the code interpreter
    """
    files = []
    path = Path(directory)

    if not path.exists():
        print(f"Directory '{directory}' does not exist, skipping file collection")
        return files

    for file_path in Path(directory).rglob("*"):
        if file_path.is_file() and not any(part.startswith(".") for part in file_path.parts):
            try:
                # Handle different file types
                if file_path.suffix.lower() in ['.csv', '.txt', '.json', '.py']:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    files.append({
                        "name": str(file_path.relative_to(directory)),
                        "encoding": "string",
                        "content": content
                    })
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    # For Excel files, we'll let pandas handle them in the code
                    print(f"Excel file detected: {file_path.name} - will be handled by pandas")

            except (UnicodeDecodeError, PermissionError) as e:
                print(f"Could not read file {file_path}: {e}")

    return files
