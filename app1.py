import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from abc import ABC, abstractmethod

# 1. SETUP
os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name="llama-3.3-70b-versatile", temperature=0)

# 2. BACKEND AGENT LOGIC
class AgentState(TypedDict):
    dataframe: object
    profile: dict
    charts: List[Any]
    insight: str
    answer: str
    user_question: str

def profile_dataset(df):
    return {"Rows": df.shape[0], "Columns": df.shape[1], "Numeric Columns": list(df.select_dtypes(include=np.number).columns)}

def analysis_agent(state):
    state["profile"] = profile_dataset(state["dataframe"])
    state["insight"] = llm.invoke("Provide 5 professional insights for this dataset: " + str(state["profile"])).content
    return state

def chat_agent(state):
    df = state["dataframe"]
    q = state["user_question"]
    ans = llm.invoke(f"Based on this data: {df.head().to_string()}. Answer this: {q}").content
    state["answer"] = ans
    return state

# Graphs
analysis_workflow = StateGraph(AgentState).add_node("a", analysis_agent).set_entry_point("a").add_edge("a", END).compile()
chat_workflow = StateGraph(AgentState).add_node("c", chat_agent).set_entry_point("c").add_edge("c", END).compile()

class UniversalAIAgent:
    def analyze(self, state): return analysis_workflow.invoke(state)
    def chat_with_data(self, state): return chat_workflow.invoke(state)

agent = UniversalAIAgent()

# 3. FRONTEND UI
st.set_page_config(page_title="Universal AI Agent", layout="wide")
page = st.sidebar.radio("Navigation", ["Home", "Dataset Preview", "AI Analysis", "Chat with Dataset", "Download Report"])
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = pd.read_csv(uploaded_file) if uploaded_file else None

if page == "Home": st.header("🏠 Welcome")
elif page == "Dataset Preview" and df is not None: st.dataframe(df)
elif page == "AI Analysis" and df is not None:
    if st.button("Run AI Analysis"):
        res = agent.analyze({"dataframe": df, "profile": {}, "charts": [], "insight": "", "answer": "", "user_question": ""})
        st.write(res["insight"])
        st.session_state["analysis_result"] = res
elif page == "Chat with Dataset" and df is not None:
    prompt = st.chat_input("Ask something...")
    if prompt:
        res = agent.chat_with_data({"dataframe": df, "user_question": prompt, "answer": "", "profile": {}, "charts": [], "insight": ""})
        st.write(res["answer"])
elif page == "Download Report" and "analysis_result" in st.session_state:
    st.download_button("Download Report", data=st.session_state["analysis_result"]["insight"], file_name="report.txt")
