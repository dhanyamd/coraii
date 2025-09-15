from corai import reasoning_model, max_iterations, temperature,  together_client,  print_boxed, print_boxed_execution_result, get_execution_summary, run_python, collect_files, together_client
from typing import Optional
import re, os
class ReActDataScienceAgent:
    def __init__(
        self,
        client,
        session_id: Optional[str] = None,
        model: str = reasoning_model,
        max_iterations: int = max_iterations
    ):
        self.client = client
        self.session_id = session_id
        self.model = model
        self.max_iterations = max_iterations

        self.system_prompt = """
        You are an expert data scientist assistant that follows the ReAct framework (Reasoning + Acting).

        CRITICAL RULES:
        1. Execute ONLY ONE action at a time - this is non-negotiable
        2. Be methodical and deliberate in your approach
        3. Always validate data before advanced analysis
        4. Never make assumptions about data structure or content
        5. Never execute potentially destructive operations without confirmation

        IMPORTANT GUIDELINES:
        - Be explorative and creative, but cautious
        - Try things incrementally and observe the results
        - Never randomly guess (e.g., column names) - always examine data first
        - If you don't have data files, use "import os; os.listdir()" to see what's available
        - When you see "Code executed successfully" or "Generated plot/image", it means your code worked
        - Plots and visualizations are automatically displayed to the user
        - Build on previous successful steps rather than starting over

        WAIT FOR THE RESULT OF THE ACTION BEFORE PROCEEDING.

        You must strictly adhere to this format (you have two options):

        ## Format 1 - For taking an action:

        Thought: Reflect on what to do next. Analyze results from previous steps. Be descriptive about your reasoning,
        what you expect to see, and how it builds on previous actions. Reference specific data points or patterns you've observed.

        Action Input:
        ```python
        <python code to run>
        ```

        ## Format 2 - ONLY when you have completely finished the task:

        Thought: Reflect on the complete process and summarize what was accomplished.

        Final Answer:
        [Provide a comprehensive summary of the analysis, key findings, and any recommendations]

        ## Example for data exploration:

        Thought: I need to start by understanding the structure and contents of the dataset. This will help me determine
        the appropriate analysis approaches. I'll load the data and examine its basic properties including shape, columns,
        data types, and a preview of the actual values.

          Action Input:
        ```python
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Load and examine the dataset
        df = pd.read_csv("data.csv")
        print(f"Dataset shape: {df.shape}")
        print(f"\\nColumn names: {df.columns.tolist()}")
        print(f"\\nData types:\\n{df.dtypes}")
        print(f"\\nFirst few rows:\\n{df.head()}")
        ```
        """
        # we will start adding the system prompt here
        self.history = [{"role": "system", "content": self.system_prompt}]

    def llm_call(self):
        """Make a call to the language model"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=temperature,
            stream=False
        )
        return response.choices[0].message.content
    def parse_response(self): 
        """Parse the LLM response and extract thought and action input"""
        response = self.llm_call() 
        if "Final Answer" in response:
            final_answer = response.split("Final Answer:")[1].strip() 
            return final_answer, None 
        if "Thought:" in response and "Action Input:" in response:
            thought = response.split("Thought:")[1].split("Action Input:")[0].strip()

            code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", response, re.DOTALL)
            if code_match:
                action_input = code_match.group(1).strip()
            else:
                print(f"ERROR: No code block found in response:\n{response}")
                raise ValueError("No code block found in the response")
        else:
            thought = "The assistant didn't follow the ReAct format properly."
            action_input = "print('Error: Format not followed by the assistant')"

        return thought, action_input
    def run(self, user_input: str):
        """Execute the main ReAct reasoning and acting loop"""
        self.history.append({"role": "user", "content": user_input})

        current_iteration = 0

        print("🚀 Starting ReAct Data Science Agent")
        print(f"📝 Task: {user_input}")
        print("=" * 80)

        while current_iteration < self.max_iterations:
            try:
                result, action_input = self.parse_response()

                if action_input is None:
                    print_boxed(result, "Final Answer", "🎯")
                    return result

                thought = result
                print_boxed(thought, f"Thought (Iteration {current_iteration + 1})", "🤔")
                print_boxed(action_input, "Action", "🛠️")

                # Execute the code
                execution_result = run_python(action_input, self.session_id)

                # Update session ID if we got a new one
                if execution_result and "session_id" in execution_result:
                    self.session_id = execution_result["session_id"]

                # Display results
                print_boxed_execution_result(execution_result, "Result", "📊")

                # Get summary for agent's history
                execution_summary = get_execution_summary(execution_result)

                # Add to conversation history. We use the "user" role for the observation content.
                # You could also use "tools".
                add_to_history = f"Thought: {thought}\nAction Input:```python\n{action_input}\n```"
                self.history.append({"role": "assistant", "content": add_to_history})
                self.history.append({"role": "user", "content": f"Observation: {execution_summary}"})

                current_iteration += 1
                print("-" * 80)

            except Exception as e:
                print(f"❌ Error in iteration {current_iteration + 1}: {str(e)}")
                # Add error to history and continue
                self.history.append({"role": "user", "content": f"Error occurred: {str(e)}. Please try a different approach."})
                current_iteration += 1

        print(f"⚠️ Maximum iterations ({self.max_iterations}) reached without completion")
        return "Task incomplete - maximum iterations reached"

 