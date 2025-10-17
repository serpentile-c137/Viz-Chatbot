import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any, TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import numpy as np

load_dotenv()

# Create a directory for charts if it doesn't exist
if not os.path.exists('charts'):
    os.makedirs('charts')

# --- App UI ---
st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Agentic Chatbot for Data Q&A and Visualization")
st.write("Upload your CSV dataset and start chatting or requesting visualizations interactively!")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- LangChain & Agent setup ---

    class BasicChatState(TypedDict):
        messages: Annotated[List[Any], add_messages]

    # Utility function to save plots
    def save_plot(fig, filename):
        filepath = os.path.join('charts', filename)
        fig.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        return filepath

    # Tool definitions with visualization logic
    @tool
    def head_tool(n: int = 5) -> str:
        """Show the first n rows of the dataset."""
        n = max(1, int(n))
        st.dataframe(df.head(n))
        return f"Displayed first {n} rows."

    @tool
    def tail_tool(n: int = 5) -> str:
        """Show the last n rows of the dataset."""
        n = max(1, int(n))
        st.dataframe(df.tail(n))
        return f"Displayed last {n} rows."

    @tool
    def describe_tool() -> str:
        """Return descriptive statistics for numeric columns and display them."""
        desc = df.describe(include='all').T
        st.dataframe(desc)
        return f"Displayed descriptive statistics for dataset ({df.shape[0]} rows, {df.shape[1]} cols)."

    @tool
    def column_stats_tool(column: str) -> str:
        """Return stats for a single column (count, mean, std, min, max, unique, missing)."""
        if column not in df.columns:
            return f"Column '{column}' not found. Available columns: {list(df.columns)}"
        series = df[column]
        stats = {
            "count": int(series.count()),
            "unique": int(series.nunique()),
            "missing": int(series.isnull().sum())
        }
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max())
            })
        st.write(stats)
        return f"Stats for column '{column}': {stats}"

    @tool
    def histogram_tool(column: str, bins: int = 10) -> str:
        """Display a histogram for a numeric column."""
        if column not in df.columns:
            return f"Column '{column}' not found. Available columns: {list(df.columns)}"
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' is not numeric."
        fig, ax = plt.subplots()
        df[column].dropna().hist(bins=max(1, int(bins)), ax=ax)
        ax.set_title(f"Histogram of {column}")
        filename = f"hist_{column}.png"
        filepath = save_plot(fig, filename)
        return f"Histogram for '{column}' displayed.", filepath

    @tool
    def correlation_tool(columns: str = "") -> str:
        """Show correlation matrix for selected columns or all numeric columns."""
        if columns:
            cols = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
            if not cols:
                return f"No valid columns provided. Available columns: {list(df.columns)}"
            corr_df = df[cols].select_dtypes(include=[np.number]).corr()
        else:
            corr_df = df.select_dtypes(include=[np.number]).corr()
        st.dataframe(corr_df)
        return "Displayed correlation matrix."

    @tool
    def filter_tool(condition: str, n: int = 10) -> str:
        """Filter rows using a pandas query condition string and display top n results."""
        try:
            res = df.query(condition)
        except Exception as e:
            return f"Query failed: {e}"
        st.dataframe(res.head(max(1, int(n))))
        return f"Displayed {min(len(res), max(1, int(n)))} rows matching condition (total matches: {len(res)})."

    @tool
    def value_counts_tool(column: str, n: int = 10) -> str:
        """Show value counts for a column and plot top n."""
        if column not in df.columns:
            return f"Column '{column}' not found. Available columns: {list(df.columns)}"
        counts = df[column].value_counts().head(max(1, int(n)))
        st.bar_chart(counts)
        return f"Displayed top {len(counts)} value counts for '{column}'."

    @tool
    def sample_tool(n: int = 5, random_state: int = 42) -> str:
        """Display a random sample of n rows."""
        n = max(1, int(n))
        res = df.sample(n=n, random_state=int(random_state))
        st.dataframe(res)
        return f"Displayed random sample of {n} rows."

    @tool
    def to_csv_tool(n: int = 100) -> str:
        """Provide a downloadable CSV sample (top n rows)."""
        n = max(1, int(n))
        sample = df.head(n)
        csv = sample.to_csv(index=False)
        st.download_button(label=f"Download top {n} rows as CSV", data=csv, file_name="sample.csv", mime="text/csv")
        return f"Provided download for top {n} rows."

    # Scatter plot tool with save visualization
    @tool
    def scatter_tool(x: str, y: str) -> str:
        """Display a scatter plot for two numerical columns."""
        if x in df.columns and y in df.columns:
            fig, ax = plt.subplots()
            ax.scatter(df[x], df[y])
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"Scatter plot of {x} vs {y}")
            filename = f"scatter_{x}_{y}.png"
            filepath = save_plot(fig, filename)
            return f"Scatter plot for '{x}' vs '{y}' displayed.", filepath
        missing = [col for col in [x, y] if col not in df.columns]
        return f"Column(s) {missing} not found. Available columns: {list(df.columns)}"

    # Pie chart tool with save visualization
    @tool
    def pie_chart_tool(column: str) -> str:
        """Display a pie chart of value distribution for a categorical column."""
        if column in df.columns:
            counts = df[column].value_counts()
            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index, autopct='%.1f%%')
            ax.set_title(f"Distribution of {column}")
            filename = f"pie_{column}.png"
            filepath = save_plot(fig, filename)
            return f"Pie chart for '{column}' displayed.", filepath
        return f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"

    @tool
    def data_info_tool() -> str:
        """Get basic info about dataset."""
        info = {
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        st.write(info)
        return f"Dataset info displayed: {info['shape']['rows']} rows, {info['shape']['cols']} columns."

    tools = [
        head_tool,
        tail_tool,
        sample_tool,
        to_csv_tool,
        describe_tool,
        data_info_tool,
        column_stats_tool,
        value_counts_tool,
        histogram_tool,
        correlation_tool,
        filter_tool,
        scatter_tool,
        pie_chart_tool,
    ]

    # Language model setup
    llm = ChatGroq(model="llama-3.1-8b-instant")
    llm_with_tools = llm.bind_tools(tools)

    # Chatbot node
    def chatbot_node(state: BasicChatState):
        """Process messages and generate responses"""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Tools router
    def tools_router(state: BasicChatState):
        """Route to tools if tool calls are present, otherwise end"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return "toolnode"
        return END

    memory = MemorySaver()
    tool_node = ToolNode(tools=tools)

    # Build graph
    graph = StateGraph(BasicChatState)
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("toolnode", tool_node)
    graph.add_conditional_edges("chatbot", tools_router)
    graph.add_edge("toolnode", "chatbot")
    graph.set_entry_point("chatbot")

    app = graph.compile(checkpointer=memory)
    thread_id = f"data_chat_session::{getattr(uploaded_file, 'name', 'unknown')}"
    config = {"configurable": {"thread_id": thread_id}}

    # --- Streamlit Chat Interface ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)
            elif isinstance(msg, AIMessage) and msg.content:
                st.chat_message("assistant").write(msg.content)

    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your message:", 
            placeholder="Ask about your data or request visualizations (e.g., 'show bar chart for column_name')"
        )
        send = st.form_submit_button("Send")

    if send and user_input.strip():
        user_message = HumanMessage(content=user_input)
        st.session_state.chat_history.append(user_message)
        system_context = HumanMessage(
            content=f"Dataset context: {df.shape[0]} rows, {df.shape[1]} cols. Columns: {', '.join(df.columns)}."
        )
        messages_for_agent = [system_context] + st.session_state.chat_history
        try:
            result = app.invoke({"messages": messages_for_agent}, config=config)
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                ai_response = ai_messages[-1]
                if ai_response.content:
                    # Check for file references in response
                    if hasattr(ai_response, 'tool_calls') and ai_response.tool_calls:
                        for call in ai_response.tool_calls:
                            if hasattr(call, "filepath") and call.filepath:
                                # Render image if it's an image or show a link if a file
                                st.image(call.filepath)
                    st.session_state.chat_history.append(ai_response)
                    st.rerun()
        except Exception as e:
            error_message = AIMessage(content=f"Sorry, I encountered an error: {str(e)}")
            st.session_state.chat_history.append(error_message)
            st.rerun()

else:
    st.info("Please upload a CSV file to start the chat.")
