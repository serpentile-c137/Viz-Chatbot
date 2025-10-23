# Import necessary libraries.
import streamlit as st
import pandas as pd
import json
import os
from agent import create_pd_agent, query_pd_agent

def decode_response(response: str) -> dict:
    return json.loads(response)

def write_response(decoded_response: dict):
    # Check if the response is an answer.
    if "answer" in decoded_response:
        st.write(decoded_response["answer"])

    # Check if the response is a bar chart.
    if "chart" in decoded_response:
        image_path = "./chart_image/chart.png"
        if os.path.exists(image_path):
            st.image(image_path)
        else:
            st.error(f"Image not found: {image_path}")

    if "table" in decoded_response:
        data = decoded_response["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

st.title("🤔 Chat with your CSV 📊")

st.write("Please upload your CSV and metadata file below.")

# Function to allow users to upload a CSV file.
csv_data = st.file_uploader("Upload your CSV file.")

# Function to allow users to input a query.
query = st.text_area("Please let me know your query.")

if st.button("Submit Query", type="primary"):
    # Create an agent from the CSV file.
    agent = create_pd_agent(csv_data)

    # Query the agent.
    response = query_pd_agent(agent=agent, query=query)

    # Decode the response.
    decoded_response = decode_response(response)

    # Write the response to the Streamlit app.
    write_response(decoded_response)