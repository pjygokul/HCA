import pandas as pd
import json
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load .env for local development
load_dotenv()

# Try Streamlit secrets first, fallback to .env
try:
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

# Initialize client
client = Groq(api_key=api_key) if api_key else None

def process_chat_query(query: str, alloc_df: pd.DataFrame) -> dict:
    """
    Parses natural language query to determine the best healthcare response and
    which geographical zone should be highlighted on the interconnected map.
    This version uses the Groq API (e.g. Llama-3) to support any free-form prompt.
    """
    if alloc_df.empty:
        return {"text": "System data is currently empty.", "highlight_zone": None}

    if not client:
        return {
            "text": "Groq API key not found. Please ensure your GROQ_API_KEY is placed in the `.env` file.", 
            "highlight_zone": None
        }

    # Format the dataframe context for the LLM
    data_context = alloc_df.to_json(orient='records')

    system_prompt = f"""You are the Chennai Flu Watch AI Assistant, an elite AI specialized in public health, disease resource allocation, and epidemiology. 
Your goal is to answer the user's question based strictly on the current zone-level data provided below.
Provide a clear, accurate, and concise answer. Do NOT guess metrics or facts not available in the data.

Rules for output:
1. You MUST respond with a valid JSON object containing exactly two keys: "text" and "highlight_zone".
2. "text": Your natural language response to the user. You can use markdown for formatting text (like **bolding** numbers or zones).
3. "highlight_zone": The exact name of the specific zone that is the primary subject of your answer, so the map can highlight it. If the answer is about multiple zones, or an overall region default, return null for this field.

Here is the current dashboard data as a JSON summary:
{data_context}
"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            model = "moonshotai/kimi-k2-instruct-0905", 
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        # Parse JSON
        result = json.loads(response.choices[0].message.content)
        
        text = result.get("text", "I couldn't generate a proper response.")
        highlight_zone = result.get("highlight_zone")
        
        # Validate highlight_zone against actual list to prevent UI bugs
        if highlight_zone and highlight_zone not in alloc_df["zone"].values:
            highlight_zone = None
            
        return {
            "text": text,
            "highlight_zone": highlight_zone
        }
    except Exception as e:
        return {"text": f"Error contacting AI: {str(e)}", "highlight_zone": None}
