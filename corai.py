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
def display_image(b64_image):
    """Display base64 encoded images from code execution results"""
    decoded_image = base64.b64decode(b64_image)
    print(data=decoded_image)
# This function creates a comprehensive summary of execution result for the model's history.
def get_execution_summary(execution_result: Dict) -> str:
    """
    Create a comprehensive summary of execution result for the model's history.
    This gives the model better context about what happened during code execution.

    Args:
        execution_result: The result dictionary from run_python

    Returns:
        A summary of the execution including status, outputs, and any errors
    """
    if not execution_result:
        return "Execution failed - no result returned"

    # Check execution status
    status = execution_result.get("status", "unknown")
    summary_parts = [f"Execution status: {status}"]

    # Process outputs
    stdout_outputs = []
    display_outputs = []
    other_outputs = []

    if "outputs" in execution_result:
        for output in execution_result["outputs"]:
            output_type = output.get("type", "unknown")
            output_data = output.get("data", "")

            if output_type == "stdout":
                stdout_outputs.append(output_data)
            elif output_type == "display_data":
                if isinstance(output_data, dict):
                    if "image/png" in output_data:
                        display_outputs.append("Generated plot/image")
                    if "text/plain" in output_data:
                        display_outputs.append(f"Display: {output_data['text/plain']}")
                else:
                    display_outputs.append("Generated display output")
            else:
                other_outputs.append(f"{output_type}: {str(output_data)[:100]}")

    # Add stdout outputs
    if stdout_outputs:
        summary_parts.append("Text output:")
        summary_parts.extend(stdout_outputs)

    # Add display outputs (plots, images)
    if display_outputs:
        summary_parts.append("Visual outputs:")
        summary_parts.extend(display_outputs)

    # Add other outputs
    if other_outputs:
        summary_parts.append("Other outputs:")
        summary_parts.extend(other_outputs)

    # Check for errors
    if "errors" in execution_result and execution_result["errors"]:
        summary_parts.append("Errors:")
        summary_parts.extend(execution_result["errors"])

    # If no outputs at all but status is success
    if not stdout_outputs and not display_outputs and not other_outputs and status == "success":
        summary_parts.append("Code executed successfully (no explicit output generated)")

    return "\n".join(summary_parts)

# This function processes the execution result and extracts text outputs and image data.
def process_execution_result(execution_result: Dict) -> tuple[str, List[str]]:
    """
    Process execution result and extract text outputs and image data.

    Args:
        execution_result: The result dictionary from run_python

    Returns:
        tuple: (text_output, list_of_image_data)
    """
    text_outputs = []
    image_data = []

    if execution_result and "outputs" in execution_result:
        for output in execution_result["outputs"]:
            if output["type"] == "stdout":
                text_outputs.append(output["data"])
            elif output["type"] == "display_data":
                # Handle display data (images, plots, etc.)
                if isinstance(output["data"], dict):
                    if "image/png" in output["data"]:
                        image_data.append(output["data"]["image/png"])
                    # Add text representation if available
                    if "text/plain" in output["data"]:
                        text_outputs.append(f"[Display Data] {output['data']['text/plain']}")

    # Join all text outputs
    combined_text = "\n".join(text_outputs) if text_outputs else ""

    return combined_text, image_data

def box_text(text: str, title: Optional[str] = None, emoji: Optional[str] = None) -> str:
    """Create a boxed text with optional title and emoji."""
    # Handle None or empty text
    if not text:
        text = "No output"

    # Limit to 500 words for readability
    words = text.split()
    if len(words) > 500:
        words = words[:500]
        words.append("...")
        text = " ".join(words)

    # Wrap text at specified width
    wrapped_lines = []
    for line in text.split("\n"):
        if len(line) > box_width:
            wrapped_lines.extend(textwrap.wrap(line, width=box_width))
        else:
            wrapped_lines.append(line)

    # Handle empty wrapped_lines
    if not wrapped_lines:
        wrapped_lines = ["No output"]

    width = max(len(line) for line in wrapped_lines)
    width = max(width, len(title) if title else 0)

    if title and emoji:
        title = f" {emoji} {title} "
    elif title:
        title = f" {title} "
    elif emoji:
        title = f" {emoji} "

    result = []
    if title:
        result.append(f"╔{'═' * (width + 2)}╗")
        result.append(f"║ {title}{' ' * (width - len(title) + 2)}║")
        result.append(f"╠{'═' * (width + 2)}╣")
    else:
        result.append(f"╔{'═' * (width + 2)}╗")

    for line in wrapped_lines:
        result.append(f"║ {line}{' ' * (width - len(line))} ║")

    result.append(f"╚{'═' * (width + 2)}╝")
    return "\n".join(result)

def print_boxed(text: str, title: Optional[str] = None, emoji: Optional[str] = None):
    """Print text in a box with optional title and emoji."""
    print(box_text(text, title, emoji))

def print_boxed_execution_result(execution_result: Dict, title: Optional[str] = None, emoji: Optional[str] = None):
    """
    Print execution result in a box and display any images as part of the output.
    """
    text_output, image_data = process_execution_result(execution_result)

    # If we have images, mention them in the text
    if image_data:
        if text_output:
            text_output += f"\n\n[Generated {len(image_data)} plot(s)/image(s) - displayed below]"
        else:
            text_output = f"[Generated {len(image_data)} plot(s)/image(s) - displayed below]"
    elif not text_output:
        text_output = "No text output"

    # Print the boxed text
    print(box_text(text_output, title, emoji))

    # Display images immediately after the box
    if show_images:
        for i, img_data in enumerate(image_data):
            if len(image_data) > 1:
                print(f"\n--- Plot/Image {i+1} ---")
            display_image(img_data)
            if i < len(image_data) - 1:
                print()