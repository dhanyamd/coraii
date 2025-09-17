from typing import Optional
from agent import ReActDataScienceAgent, run_python, collect_files
from corai import together_client 
import os 
def create_agent_with_data(data_dir: Optional[str] = None) -> ReActDataScienceAgent:
    """
    Create a ReAct agent with optional data file upload

    Args:
        query: The data science task to perform
        data_dir: Optional directory containing data files to upload

    Returns:
        Configured ReAct agent ready to run
    """
    session_id = None

    # Handle file uploads if data directory provided
    if data_dir and os.path.exists(data_dir):
        print(f"📁 Collecting files from {data_dir}...")
        files = collect_files(data_dir)

        if files:
            print(f"📤 Found {len(files)} files. Initializing session with uploaded files...")

            # Initialize session with files
            init_result = run_python("print('Session initialized with data files')", None, files)

            if init_result and "session_id" in init_result:
                session_id = init_result["session_id"]
                print(f"✅ Session initialized with ID: {session_id}")
            else:
                print("⚠️ Failed to get session ID, continuing without persistent session")
        else:
            print("📂 No valid files found in directory")

    # Create and return the agent
    agent = ReActDataScienceAgent(
        client=together_client,
        session_id=session_id
    )

    return agent

def run_data_science_task(query: str, data_dir: Optional[str] = None) -> str:
    """
    Convenience function to run a complete data science task

    Args:
        query: The data science task description
        data_dir: Optional directory with data files

    Returns:
        The final result from the agent
    """
    agent = create_agent_with_data(data_dir)
    return agent.run(query) 

# Example 1: Simple iris dataset visualization
query_1 = "Load the iris dataset and create a scatter plot of sepal length vs sepal width, colored by species"

print("=" * 80)
print("🌸 EXAMPLE 1: Iris Dataset Visualization")
print("=" * 80)
print(f"Task: {query_1}")
print()

# Run the agent
result_1 = run_data_science_task(query_1)