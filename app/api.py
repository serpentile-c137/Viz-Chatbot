from fastapi import FastAPI, Query
from typing import Optional
# from app.main import ask_agent_with_plot
from main import ask_agent_with_plot
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Ensure GROQ_API_KEY is set
if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError("GROQ_API_KEY is not set in the environment variables.")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Viz-Chatbot API!"}

@app.get("/ask")
def ask_query(query: str):
    """
    Endpoint to query the agent.
    Example: /ask?query=Create a histogram of the column age with 20 bins.
    """
    text, plot_path = ask_agent_with_plot(query)
    response = {"response": text}
    if plot_path:
        response["plot_path"] = plot_path
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
