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
    def bar_chart_tool(column: str) -> str:
        """Display a bar chart of value counts for the given column in the uploaded dataframe.
        
        Args:
            column: The column name to create a bar chart for
            
        Returns:
            A confirmation message that the chart was displayed
        """
        if column in df.columns:
            st.bar_chart(df[column].value_counts())
            return f"Bar chart for '{column}' displayed successfully."
        else:
            return f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"

    @tool
    def scatter_tool(x: str, y: str) -> str:
        """Display a scatter plot using the given x and y columns from the uploaded dataframe.
        
        Args:
            x: The column name for x-axis
            y: The column name for y-axis
            
        Returns:
            A confirmation message that the scatter plot was displayed
        """
        if x in df.columns and y in df.columns:
            st.scatter_chart(df[[x, y]])
            return f"Scatter plot for '{x}' vs '{y}' displayed successfully."
        else:
            missing = [col for col in [x, y] if col not in df.columns]
            return f"Column(s) {missing} not found. Available columns: {list(df.columns)}"

    @tool
    def pie_chart_tool(column: str) -> str:
        """Display a pie chart of value distribution for the given column in the uploaded dataframe.
        
        Args:
            column: The column name to create a pie chart for
            
        Returns:
            A confirmation message that the pie chart was displayed
        """
        if column in df.columns:
            counts = df[column].value_counts()
            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index, autopct='%.1f%%')
            ax.set_title(f"Distribution of {column}")
            st.pyplot(fig)
            plt.close()
            return f"Pie chart for '{column}' displayed successfully."
        else:
            return f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"

    @tool
    def data_info_tool() -> str:
        """Get basic information about the uploaded dataset including shape, columns, and data types.
        
        Returns:
            A summary of the dataset information
        """
        info = f"""Dataset Information:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Columns: {', '.join(df.columns.tolist())}
        - Data types: {df.dtypes.to_dict()}
        - Missing values: {df.isnull().sum().to_dict()}
        """
        return info

    tools = [
        bar_chart_tool,
        scatter_tool,
        pie_chart_tool,
        data_info_tool,
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
