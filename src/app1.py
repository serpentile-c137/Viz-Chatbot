import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

    # Tool definitions using @tool decorator with proper docstrings
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
        st.pyplot(fig)
        plt.close()
        return f"Histogram for '{column}' displayed."

    @tool
    def correlation_tool(columns: str = "") -> str:
        """Show correlation matrix. Provide comma-separated columns or leave empty for all numeric cols."""
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
        """Filter rows using a pandas query condition string and display top n results.
        Example condition: \"age > 30 and country == 'US'\""""
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
        """Provide a downloadable CSV sample (top n rows). Returns a link-like message."""
        n = max(1, int(n))
        sample = df.head(n)
        csv = sample.to_csv(index=False)
        st.download_button(label=f"Download top {n} rows as CSV", data=csv, file_name="sample.csv", mime="text/csv")
        return f"Provided download for top {n} rows."

    # keep scatter and pie utilities as specialized visual tools
    @tool
    def scatter_tool(x: str, y: str) -> str:
        """Display a scatter plot using the given x and y columns from the uploaded dataframe."""
        if x in df.columns and y in df.columns:
            st.scatter_chart(df[[x, y]])
            return f"Scatter plot for '{x}' vs '{y}' displayed successfully."
        missing = [col for col in [x, y] if col not in df.columns]
        return f"Column(s) {missing} not found. Available columns: {list(df.columns)}"

    @tool
    def pie_chart_tool(column: str) -> str:
        """Display a pie chart of value distribution for the given column in the uploaded dataframe."""
        if column in df.columns:
            counts = df[column].value_counts()
            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index, autopct='%.1f%%')
            ax.set_title(f"Distribution of {column}")
            st.pyplot(fig)
            plt.close()
            return f"Pie chart for '{column}' displayed successfully."
        return f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"

    @tool
    def data_info_tool() -> str:
        """Get basic information about the uploaded dataset including shape, columns, and data types."""
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

    llm = ChatGroq(model="llama-3.1-8b-instant")
    llm_with_tools = llm.bind_tools(tools)

    # Define the chatbot node function
    def chatbot_node(state: BasicChatState):
        """Process messages and generate responses"""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Define the tools router
    def tools_router(state: BasicChatState):
        """Route to tools if tool calls are present, otherwise end"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return "toolnode"
        return END

    memory = MemorySaver()
    tool_node = ToolNode(tools=tools)

    # Build the graph
    graph = StateGraph(BasicChatState)
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("toolnode", tool_node)

    graph.add_conditional_edges("chatbot", tools_router)
    graph.add_edge("toolnode", "chatbot")
    graph.set_entry_point("chatbot")

    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "data_chat_session"}}

    # --- Chat Interface ---
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
        # Add user message to chat history
        user_message = HumanMessage(content=user_input)
        st.session_state.chat_history.append(user_message)
        
        # Prepare messages for the agent (including context about the dataset)
        context_message = HumanMessage(
            content=f"You are helping analyze a dataset with {df.shape[0]} rows and {df.shape[1]} columns. "
            f"Available columns: {', '.join(df.columns)}. "
            f"Please help the user with their question: {user_input}"
        )
        
        # Invoke the agent
        try:
            result = app.invoke({"messages": [context_message]}, config=config)
            
            # Get the last AI message from the result
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                ai_response = ai_messages[-1]
                if ai_response.content:
                    st.session_state.chat_history.append(ai_response)
                    st.rerun()  # Refresh to show the new message
        except Exception as e:
            error_message = AIMessage(content=f"Sorry, I encountered an error: {str(e)}")
            st.session_state.chat_history.append(error_message)
            st.rerun()

else:
    st.info("Please upload a CSV file to start the chat.")
