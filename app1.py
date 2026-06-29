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

# Set API Key directly for testing (Use environment variables for production)
os.environ["GROQ_API_KEY"] = "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"

# Initialize LLM
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# ==========================================
# 1. AI AGENT BACKEND LOGIC (LANGGRAPH)
# ==========================================

# Define Unified Agent State
class AgentState(TypedDict):
    dataframe: object
    profile: dict
    charts: List[Any]
    query_plan: Dict[str, Any]
    query_result: Any
    insight: str
    answer: str
    errors: List[str]
    memory: List[str]
    user_question: str
    tool_decision: Dict[str, Any]

# Dataset Profile Function
def profile_dataset(df):
    profile = {}
    profile["Rows"] = df.shape[0]
    profile["Columns"] = df.shape[1]
    profile["Column Names"] = list(df.columns)
    profile["Data Types"] = df.dtypes.astype(str).to_dict()
    profile["Missing Values"] = df.isnull().sum().to_dict()
    profile["Duplicate Rows"] = int(df.duplicated().sum())
    profile["Memory Usage (MB)"] = round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    profile["Numeric Columns"] = list(df.select_dtypes(include=np.number).columns)
    profile["Categorical Columns"] = list(df.select_dtypes(include=["object","category"]).columns)
    return profile

# Base Tool Structure
class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self): pass
    @property
    @abstractmethod
    def description(self): pass
    @abstractmethod
    def execute(self, df, params): pass

# Visualization Tools Classes
class HistogramTool(BaseTool):
    @property
    def name(self): return "histogram"
    @property
    def description(self): return "Create histogram for numeric column"
    def execute(self, df, params):
        column = params["column"]
        return px.histogram(df, x=column, title=f"Distribution of {column}")

class HeatmapTool(BaseTool):
    @property
    def name(self): return "heatmap"
    @property
    def description(self): return "Correlation heatmap"
    def execute(self, df, params):
        corr = df.select_dtypes(include=np.number).corr()
        return px.imshow(corr, text_auto=True, title="Correlation Matrix")

class BarChartTool(BaseTool):
    @property
    def name(self): return "bar_chart"
    @property
    def description(self): return "Category vs Numeric Analysis"
    def execute(self, df, params):
        cat = params["category"]
        num = params["value"]
        grouped = df.groupby(cat)[num].sum().reset_index()
        return px.bar(grouped, x=cat, y=num, title=f"{num} by {cat}")

TOOL_REGISTRY = {
    "histogram": HistogramTool(),
    "heatmap": HeatmapTool(),
    "bar_chart": BarChartTool()
}

# Agent Node Functions
def profiling_agent(state):
    state["profile"] = profile_dataset(state["dataframe"])
    return state

def tool_planner_agent(state):
    profile = state["profile"]
    available_tools = [{"name": t.name, "description": t.description} for t in TOOL_REGISTRY.values()]
    prompt = f"You are an expert Data Analyst. Select tools.\nProfile: {profile}\nTools: {available_tools}\nReturn ONLY valid JSON format like: {{\"tools\":[{{\"name\":\"histogram\",\"params\": {{\"column\":\"Sales_Amount\"}}}}]}}"
    
    response = llm.invoke(prompt)
    
    # Fix for JSON Decode Error
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        state["tool_decision"] = json.loads(content)
    except Exception as e:
        state["tool_decision"] = {"tools":[]}
        state.setdefault("errors", []).append(f"JSON Error: {str(e)}")
        
    return state

def robust_tool_executor(state):
    df = state["dataframe"]
    decisions = state.get("tool_decision", {})
    generated_charts = []
    
    if decisions and "tools" in decisions:
        for tool_info in decisions["tools"]:
            try:
                tool_name = tool_info["name"]
                params = tool_info.get("params", {})
                if tool_name in TOOL_REGISTRY:
                    fig = TOOL_REGISTRY[tool_name].execute(df, params)
                    generated_charts.append(fig)
            except Exception as e:
                pass
                
    state["charts"] = generated_charts
    return state

def insight_agent(state):
    prompt = f"You are a Senior Business Analyst.\nProfile: {state['profile']}\nCharts Generated: {len(state['charts'])}\nGenerate a professional report covering: 1. Executive Summary 2. Data Quality 3. Key Insights 4. Business Risks 5. Recommendations."
    response = llm.invoke(prompt)
    state["insight"] = response.content
    return state

# Compile LangGraph Workflows
analysis_graph = StateGraph(AgentState)
analysis_graph.add_node("profiling", profiling_agent)
analysis_graph.add_node("planner", tool_planner_agent)
analysis_graph.add_node("executor", robust_tool_executor)
analysis_graph.add_node("insight", insight_agent)

analysis_graph.set_entry_point("profiling")
analysis_graph.add_edge("profiling", "planner")
analysis_graph.add_edge("planner", "executor")
analysis_graph.add_edge("executor", "insight")
analysis_graph.add_edge("insight", END)
analysis_workflow = analysis_graph.compile()

# Master Agent Class
class UniversalAIAgent:
    def analyze(self, state):
        return analysis_workflow.invoke(state)

# Initialize Agent
agent = UniversalAIAgent()


# ==========================================
# 2. STREAMLIT FRONTEND UI
# ==========================================

st.set_page_config(
    page_title="Universal AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.main{
    background-color:#0E1117;
}
.block-container{
    padding-top:1rem;
}
h1,h2,h3{
    color:white;
}
.stButton>button{
    width:100%;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Universal AI Data Analyst Agent")
st.write("Welcome to the Universal AI Data Analyst Agent. Upload any CSV, Excel, or JSON dataset and analyze it.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Dataset Preview",
        "Profiling",
        "Visualization",
        "AI Analysis",
        "Chat with Dataset",
        "Download Report"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv","xlsx","json"]
)

# Dataset Loader
df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        st.sidebar.success("Dataset Loaded Successfully")
    except Exception as e:
        st.sidebar.error(str(e))

# Page: Home
if page == "Home":
    st.header("🏠 Home")
    st.info("Features:\n✅ Upload Dataset\n✅ Dataset Preview\n✅ Profiling\n✅ Visualization\n✅ AI Analysis\n✅ Chat with Dataset\n✅ Report Generation")

# Page: AI Analysis (Focus based on your screenshot)
elif page == "AI Analysis":
    st.header("🤖 AI Dataset Analysis")

    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        st.success("Dataset loaded successfully.")

        if st.button("Run AI Analysis"):
            with st.spinner("Analyzing dataset... Please wait."):
                try:
                    # Setup initial state
                    state = {
                        "dataframe": df,
                        "profile": {},
                        "charts": [],
                        "query_plan": {},
                        "query_result": "",
                        "tool_decision": {},
                        "insight": "",
                        "answer": "",
                        "errors": [],
                        "memory": [],
                        "user_question": "Analyze this dataset."
                    }

                    # Trigger AI Agent
                    result = agent.analyze(state)
                    
                    # Store result in session state for later use
                    st.session_state["analysis_result"] = result

                    st.success("Analysis Completed")
                    
                    # Display Dataset Summary Table exactly like your screenshot
                    st.subheader("Dataset Summary")
                    summary = pd.DataFrame({
                        "Property": ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
                        "Value": [df.shape[0], df.shape[1], df.isnull().sum().sum(), df.duplicated().sum()]
                    })
                    st.table(summary)

                    # Display Detailed Profile Dataframe
                    st.subheader("Dataset Profile")
                    profile_df = pd.DataFrame({
                        "Column": df.columns,
                        "Data Type": df.dtypes.astype(str),
                        "Missing": df.isnull().sum().values,
                        "Unique": df.nunique().values
                    })
                    st.dataframe(profile_df, use_container_width=True)

                    # Display Generated Charts using Plotly
                    st.subheader("Generated Charts")
                    charts = result.get("charts", [])
                    if charts:
                        for chart in charts:
                            try:
                                # Plotly charts rendering
                                st.plotly_chart(chart, use_container_width=True)
                            except Exception:
                                pass
                    else:
                        st.info("No charts were generated by the AI for this specific dataset.")

                    # Display AI Insights
                    st.subheader("AI Business Insights")
                    st.markdown(result.get("insight", "No insights generated."))

                except Exception as e:
                    st.error(f"Analysis failed due to an error: {e}")

# Page: Placeholder for others to prevent errors
elif page in ["Dataset Preview", "Profiling", "Visualization", "Chat with Dataset", "Download Report"]:
    st.header(page)
    st.info(f"{page} module is under development or requires you to upload data first.")
