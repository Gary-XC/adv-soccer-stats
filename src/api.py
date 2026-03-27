from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from langchain_ollama import OllamaLLM
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# custom tools
from langchain_core.tools import Tool, tool
from rapidfuzz import process, fuzz
from src.vis_fn import plot_pizza_chart, plot_usage_scatter


app = FastAPI(title="Soccer Analytics AI Engine")

# Dynamically find the data path regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_analytics.csv")

# --- MEMORY MANAGEMENT ---
# Dictionary to hold conversational memory for different users/sessions
sessions_memory = {}


def get_memory_string(session_id: str) -> str:
    """Retrieves and formats the chat history for a session."""
    if session_id not in sessions_memory:
        sessions_memory[session_id] = []

    history = sessions_memory[session_id]
    if not history:
        return ""

    # Format the last few interactions into a clean text block
    formatted_history = "\n".join([f"User: {q}\nAgent: {a}" for q, a in history])
    return formatted_history


def save_to_memory(session_id: str, question: str, answer: str):
    """Saves the interaction and enforces the sliding window (k=5)."""
    if session_id not in sessions_memory:
        sessions_memory[session_id] = []

    sessions_memory[session_id].append((question, answer))

    # The "Window Buffer" logic: Keep only the last 5 interactions
    if len(sessions_memory[session_id]) > 5:
        sessions_memory[session_id].pop(0)


@tool
def get_closest_player_name(user_input: str) -> str:
    """
    Use this tool when you cannot find a player name in the dataset or suspect a typo.
    It returns the closest matching player name from the official database.
    """
    # Grab the list directly from the global 'df' instead of an argument
    player_list = df["Player"].unique().tolist()

    # Perform the fuzzy match
    match, score, index = process.extractOne(user_input, player_list, scorer=fuzz.WRatio)  # type: ignore

    if score > 80:
        return f"The correct name in the database is likely '{match}'."
    return "No close match found. Please ask the user for clarification."


# -------------------------

# Initialize the agent globally
try:
    df = pd.read_csv(DATA_PATH)
    llm = OllamaLLM(model="llama3.1", temperature=0)

    SCOUT_PREFIX = """
    You are an elite European football data scout and advanced analytics expert. 
    Your goal is to answer questions using ONLY the provided pandas dataframe. 
    
    STRICT RULES:
    1. Tone: Professional, analytical, and concise. Use proper soccer terminology.
    2. Grounding: Do not hallucinate external stats. Base your analysis entirely on the data provided.
    3. FORMATTING: You must ALWAYS start your final response with "Final Answer: ".
    4. TOOL USAGE: When using the python_repl_ast tool, you MUST output EXACTLY "Action: python_repl_ast" and nothing else on that line. Do not add comments or explanations to the Action line.
    5. THE FOLLOW-UP MANDATE: You must ALWAYS conclude your final answer by suggesting exactly ONE insightful, data-driven follow-up question the user could ask to dive deeper into the analysis or compare players. Format this exactly as: 
    "Suggested Follow-up: [Your question]"
    6. ERROR RECOVERY & TOOL PRIORITY: 
    - If a tool returns an error saying a player was not found, IMMEDIATELY call 'get_closest_player_name'. 
    - Once 'get_closest_player_name' returns a suggestion, you MUST trust it. 
    - DO NOT use 'python_repl_ast' to search for the name again. 
    - Proceed IMMEDIATELY to use the corrected name with the visualization tool.
    """

    custom_tools = [
        get_closest_player_name,
        # Tool for Individual Player Analysis
        Tool(
            name="GeneratePizzaChart",
            func=lambda name: plot_pizza_chart(name, df),
            description="""Use this ONLY when the user asks for a visual comparison, 
        percentile chart, or 'pizza chart' for a specific player. 
        Input should be the EXACT player name from the database.""",
        ),
        # Tool for League-Wide Analysis
        Tool(
            name="GenerateUsageScatter",
            func=lambda _: plot_usage_scatter(df),
            description="""Use this when the user wants to see the big picture, 
        macro view, or a scatter plot of the whole league's usage vs efficiency.""",
        ),
    ]

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        prefix=SCOUT_PREFIX,
        extra_tools=custom_tools,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,  # Prevents the 500 crashes
    )
    print("AI Agent successfully loaded into memory.")
except Exception as e:
    print(f"Failed to initialize agent: {e}")
    agent = None


class QueryRequest(BaseModel):
    question: str
    session_id: str = (
        "default_user"  # Added so you can easily scale to multiple users later
    )


@app.get("/health")
def health_check():
    return {"status": "online", "agent_loaded": agent is not None}


@app.post("/ask")
def ask_ai(request: QueryRequest):
    if agent is None:
        raise HTTPException(status_code=500, detail="AI Agent is not initialized.")

    # 1. Retrieve this user's memory string
    chat_history_str = get_memory_string(request.session_id)

    # 2. Inject the memory into the prompt if history exists
    if chat_history_str:
        augmented_question = f"Previous Conversation Context:\n{chat_history_str}\n\nNew User Question: {request.question}"
    else:
        augmented_question = request.question

    try:
        # 3. Ask the LLM
        response = agent.invoke(augmented_question)  # type: ignore
        final_answer = response["output"]

        # 4. Save this specific interaction to our custom memory buffer
        save_to_memory(request.session_id, request.question, final_answer)

        return {"answer": final_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
