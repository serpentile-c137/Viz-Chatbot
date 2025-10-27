# Viz-Chatbot

## Running with Docker
1. Build the Docker image:
   ```bash
   docker build -t viz-chatbot .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 viz-chatbot
   ```

## Running with Streamlit
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run app/app.py
   ```