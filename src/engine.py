import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


def run_local_agent():
    print("Waking up local LLM (Llama 3.1) in WSL...")

    try:
        df = pd.read_csv("../data/processed/master_analytics.csv")
    except FileNotFoundError:
        print("Error: master_analytics.csv not found. Make sure main.py was ran first")
        return

    llm = OllamaLLM(model="llama3.1", temperature=0)

    # --- THE SYSTEM PROMPT: Forcing Persona & Rules ---
    SCOUT_PREFIX = """
    You are an elite European football data scout and advanced analytics expert. 
    Your goal is to answer questions using ONLY the provided pandas dataframe. 
    
    STRICT RULES:
    1. Tone: Professional, analytical, and concise. Use proper soccer terminology.
    2. Grounding: Do not hallucinate external stats. Base your analysis entirely on the data provided.
    3. FORMATTING: You must ALWAYS start your final response with "Final Answer: ".
    4. THE FOLLOW-UP MANDATE: You must ALWAYS conclude your final answer by suggesting exactly ONE insightful, data-driven follow-up question the user could ask to dive deeper into the analysis or compare players. Format this exactly as: 
    "Suggested Follow-up: [Your question]"
    """
    # ------------------------------------------------------

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        prefix=SCOUT_PREFIX,  # Injecting the persona here
        verbose=True,
        allow_dangerous_code=True,
    )

    question = "Which 3 players have the highest Attacking_USG_pct? Please just give me their names and the percentage."
    print(f"\n You: {question}")

    try:
        response = agent.invoke(question)  # type: ignore
        print("\n Final Answer:")
        print(response["output"])
    except Exception as e:
        print(f"\n The agent encountered an error: {e}")


if __name__ == "__main__":
    run_local_agent()
