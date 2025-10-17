# agentic_chatbot_app.py

import streamlit as st
import pandas as pd
import io
from typing import List, Dict, Any, TypedDict
from dotenv import load_dotenv
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

load_dotenv()

# --- Visualization tools ---
def make_bar_chart(df: pd.DataFrame, column: str):
    st.bar_chart(df[column].value_counts())

def make_scatter_plot(df: pd.DataFrame, x: str, y: str):
    st.scatter_chart(df[[x, y]])

def make_pie_chart(df: pd.DataFrame, column: str):
    counts = df[column].value_counts()
    st.write(f"Distribution of {column}:")
    st.pyplot(counts.plot.pie(autopct="%.1f%%").get_figure())

# --- File upload and memory ---
st.title("Agentic Chatbot for Data Q&A and Visualization")
st.write("Upload your CSV dataset and start asking questions about your data!")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # --- LangChain & Agent setup ---
    class BasicChatState(TypedDict):
        messages: List[Any]

    # Tool definitions for visualization
    def bar_chart_tool(column: str) -> str:
        make_bar_chart(df, column)
        return f"Bar chart for '{column}' displayed."

    def scatter_tool(x: str, y: str) -> str:
        make_scatter_plot(df, x, y)
        return f"Scatterplot for '{x}' vs '{y}' displayed."

    def pie_chart_tool(column: str) -> str:
        make_pie_chart(df, column)
        return f"Pie chart for '{column}' displayed."

    # Agent tools registry
    tools = [
        {"name": "bar_chart", "description": "Plot a bar chart of a column", "func": bar_chart_tool},
        {"name": "scatter_plot", "description": "Plot a scatterplot of two columns", "func": scatter_tool},
        {"name": "pie_chart", "description": "Plot a pie chart of a column", "func": pie_chart_tool},
    ]

    llm = ChatGroq(model="llama-3-8b-instant")
    llm_with_tools = llm.bind_tools(tools)

    memory = MemorySaver()
    tool_node = ToolNode(tools=tools)
    graph = StateGraph(BasicChatState)
    graph.add_node("chatbot", lambda state: llm_with_tools.invoke(state["messages"]))
    graph.add_node("toolnode", tool_node)

    # Flow control logic reused from reference
    def tools_router(state: BasicChatState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return "toolnode"
        return END

    graph.add_conditional_edges("chatbot", tools_router)
    graph.add_edge("toolnode", "chatbot")
    graph.set_entry_point("chatbot")
    app = graph.compile(checkpointer=memory, thread_id="1")

    # --- Interactive chat loop ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question or request a visualization (e.g., 'Show a scatterplot for salary vs age'):")

    if user_input:
        # Streamlit: maintain history for conversational context
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        result = app.invoke({"messages": st.session_state.chat_history}, config={})
        response = result["messages"][-1].content
        st.session_state.chat_history.append(AIMessage(content=response))
        st.markdown(f"**AI:** {response}")
