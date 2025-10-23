# Import necessary libraries.
# import openai
# from langchain.llms import AzureOpenAI
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv
import os
from langchain.tools import Tool

load_dotenv()

def create_pd_agent(dataframe, llm=None):
    """
    Custom wrapper for creating a pandas dataframe agent.
    """
    if llm is None:
        llm = ChatGroq(model='llama-3.3-70b-versatile')

    # Define a tool for interacting with the dataframe
    def query_dataframe(query: str) -> str:
        # Implement your logic for querying the dataframe here
        return "Query result placeholder"

    tools = [
        Tool(
            name="DataframeQueryTool",
            func=query_dataframe,
            description="Use this tool to query the dataframe."
        )
    ]

    return initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Define a function to query the agent.
def query_pd_agent(agent, query):
    prompt = (
        """
        You must need to use matplotlib library if required to create a any chart.

        If the query requires creating a chart, please save the chart as "./chart_image/chart.png" and "Here is the chart:" when reply as follows:
        {"chart": "Here is the chart:"}

        If the query requires creating a table, reply as follows:
        {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
        
        If the query is just not asking for a chart, but requires a response, reply as follows:
        {"answer": "answer"}
        Example:
        {"answer": "The product with the highest sales is 'Minions'."}
        
        Lets think step by step.

        Here is the query: 
        """
        + query
    )

    # Run the agent with the prompt.
    response = agent.run(prompt)

    # Return the response in string format.
    return response.__str__()