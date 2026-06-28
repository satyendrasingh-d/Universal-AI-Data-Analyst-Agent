import streamlit as st
import pandas as pd
import numpy as np
import json

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
st.write("""
Welcome to the Universal AI Data Analyst Agent.
Upload any CSV, Excel, or JSON dataset and analyze it.
""")
# ---------------- Sidebar ----------------
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
# ---------------- Dataset Loader ----------------
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
# ---------------- Home ----------------
if page=="Home":
    st.header("🏠 Home")
    st.info("""
Features

✅ Upload Dataset

✅ Dataset Preview

✅ Profiling

✅ Visualization

✅ AI Analysis

✅ Chat with Dataset

✅ Report Generation

""")

# ---------------- Dataset Preview ----------------
elif page=="Dataset Preview":
    if df is None:
        st.warning("Please Upload Dataset First")
    else:
        st.header("📊 Dataset Preview")
        st.dataframe(df,use_container_width=True)
        col1,col2,col3=st.columns(3)
        col1.metric("Rows",df.shape[0])
        col2.metric("Columns",df.shape[1])
        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )
        st.divider()
        st.subheader("Column Types")
        st.dataframe(
            pd.DataFrame(
                {
                    "Column":df.columns,
                    "Datatype":df.dtypes.astype(str)
                }
            ),
            use_container_width=True
        )
        st.divider()
        st.subheader("Missing Values")
        missing=pd.DataFrame({
            "Column":df.columns,
            "Missing":df.isnull().sum()
        })
        st.dataframe(
            missing,
            use_container_width=True
        )
        st.divider()
        st.subheader("Descriptive Statistics")
        st.dataframe(
            df.describe(include="all"),
            use_container_width=True
        )
# ---------------- Placeholder Pages ----------------
elif page=="Profiling":
    if df is None:
        st.warning("Please upload a dataset.")
    else:
        st.header("Dataset Profiling")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Rows",df.shape[0])
        c2.metric("Columns",df.shape[1])
        c3.metric("Missing",int(df.isnull().sum().sum()))
        c4.metric("Duplicates",int(df.duplicated().sum()))
        st.subheader("Data Types")
        st.dataframe(pd.DataFrame({"Column":df.columns,"Type":df.dtypes.astype(str)}),use_container_width=True)
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={"index":"Column",0:"Missing"}),use_container_width=True)
        st.subheader("Statistics")
        st.dataframe(df.describe(include="all"),use_container_width=True)
elif page=="Visualization":
    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        st.header("📉 Data Visualization")
        numeric_columns=df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns=df.select_dtypes(include=["object","category"]).columns.tolist()

        chart_type=st.selectbox(
            "Select Chart",
            ["Histogram","Line Chart","Bar Chart","Scatter Plot","Box Plot","Pie Chart"]
        )

        if chart_type=="Histogram":
            if numeric_columns:
                col=st.selectbox("Select Numeric Column",numeric_columns)
                st.bar_chart(df[col].value_counts().sort_index())
            else:
                st.warning("No numeric columns found.")

        elif chart_type=="Line Chart":
            if numeric_columns:
                col=st.selectbox("Select Numeric Column",numeric_columns,key="line")
                st.line_chart(df[col])
            else:
                st.warning("No numeric columns found.")

        elif chart_type=="Bar Chart":
            if categorical_columns and numeric_columns:
                x=st.selectbox("Category Column",categorical_columns)
                y=st.selectbox("Value Column",numeric_columns)
                chart_df=df.groupby(x)[y].sum()
                st.bar_chart(chart_df)
            else:
                st.warning("Suitable columns not found.")

        elif chart_type=="Scatter Plot":
            if len(numeric_columns)>=2:
                x=st.selectbox("X Axis",numeric_columns,key="x")
                y=st.selectbox("Y Axis",numeric_columns,key="y")
                st.scatter_chart(df[[x,y]])
            else:
                st.warning("At least two numeric columns are required.")

        elif chart_type=="Box Plot":
            if numeric_columns:
                col=st.selectbox("Select Numeric Column",numeric_columns,key="box")
                st.write(df[col].describe())
            else:
                st.warning("No numeric columns found.")

        elif chart_type=="Pie Chart":
            if categorical_columns:
                col=st.selectbox("Select Category Column",categorical_columns,key="pie")
                st.write(df[col].value_counts())
            else:
                st.warning("No categorical columns found.")
    st.header("🤖 AI Analysis")
    st.info("Part 2 me LangGraph + Groq Analysis connect karenge.")
elif page=="Chat with Dataset":
    st.header("💬 Chat with Dataset")
    st.info("Part 3 me AI Chat connect karenge.")
elif page=="Download Report":
    st.header("📄 Download Report")
    st.info("Part 3 me PDF Report connect karenge.")

elif page=="Profiling":
    if df is None:
        st.warning("Please upload a dataset.")
    else:
        st.header("Dataset Profiling")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Rows",df.shape[0])
        c2.metric("Columns",df.shape[1])
        c3.metric("Missing",int(df.isnull().sum().sum()))
        c4.metric("Duplicates",int(df.duplicated().sum()))
        st.subheader("Data Types")
        st.dataframe(pd.DataFrame({"Column":df.columns,"Type":df.dtypes.astype(str)}),use_container_width=True)
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={"index":"Column",0:"Missing"}),use_container_width=True)
        st.subheader("Statistics")
        st.dataframe(df.describe(include="all"),use_container_width=True)


elif page=="Visualization":
    if df is None:
        st.warning("Please upload a dataset.")
    else:
        st.header("Visualization")
        numeric=df.select_dtypes(include=np.number).columns.tolist()
        categorical=df.select_dtypes(include=["object","category"]).columns.tolist()
        chart=st.selectbox("Chart Type",["Histogram","Box Plot","Scatter Plot","Bar Chart","Line Chart"])
        if chart=="Histogram":
            col=st.selectbox("Numeric Column",numeric,key="hist")
            st.bar_chart(df[col].value_counts().sort_index())
        elif chart=="Box Plot":
            col=st.selectbox("Numeric Column",numeric,key="box")
            st.write(df[[col]].describe())
            st.plotly_chart(px.box(df,y=col),use_container_width=True)
        elif chart=="Scatter Plot":
            x=st.selectbox("X Axis",numeric,key="x")
            y=st.selectbox("Y Axis",numeric,key="y")
            st.plotly_chart(px.scatter(df,x=x,y=y),use_container_width=True)
        elif chart=="Bar Chart":
            x=st.selectbox("Category",categorical,key="barx")
            y=st.selectbox("Value",numeric,key="bary")
            temp=df.groupby(x)[y].sum().reset_index()
            st.plotly_chart(px.bar(temp,x=x,y=y),use_container_width=True)
        elif chart=="Line Chart":
            col=st.selectbox("Numeric Column",numeric,key="line")
            st.line_chart(df[col])

elif page=="AI Analysis":
    st.header("AI Dataset Analysis")
    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        if st.button("Run AI Analysis"):
            with st.spinner("Analyzing dataset..."):
                try:
                    state={
                        "dataframe":df,
                        "profile":{},
                        "charts":[],
                        "query_plan":{},
                        "query_result":"",
                        "tool_decision":{},
                        "insight":"",
                        "answer":"",
                        "errors":[],
                        "memory":[],
                        "user_question":"Analyze this dataset."
                    }

                    if "analysis_workflow" in globals():
                        result=analysis_workflow.invoke(state)
                    elif "agent" in globals():
                        result=agent.analyze(state)
                    else:
                        result=state
                    st.success("Analysis Completed")
                    if "profile" in result:
                        st.subheader("Dataset Profile")
                        st.write(result["profile"])
                    if "insight" in result:
                        st.subheader("AI Insights")
                        st.write(result["insight"])
                    if "charts" in result:
                        st.subheader("Generated Charts")
                        for chart in result["charts"]:
                            try:
                                st.image(chart,use_container_width=True)
                            except:
                                st.write(chart)
                    st.session_state["analysis_result"]=result
                except Exception as e:
                    st.error(str(e))
elif page=="Chat with Dataset":
    st.header("Chat with Dataset")
    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages=[]
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        prompt=st.chat_input("Ask anything about your dataset")
        if prompt:
            st.session_state.messages.append(
                {
                    "role":"user",
                    "content":prompt
                }
            )
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        state={
                            "dataframe":df,
                            "profile":{},
                            "charts":[],
                            "query_plan":{},
                            "query_result":"",
                            "tool_decision":{},
                            "insight":"",
                            "answer":"",
                            "errors":[],
                            "memory":[],
                            "user_question":prompt
                        }
                        if "chat_workflow" in globals():
                            result=chat_workflow.invoke(state)
                            answer=result.get("answer","No answer generated.")
                        elif "agent" in globals():
                            result=agent.chat_with_data(state)
                            answer=result.get("answer","No answer generated.")
                        else:
                            answer="Chat workflow not found."
                    except Exception as e:
                        answer=str(e)
                    st.markdown(answer)
            st.session_state.messages.append(
                {
                    "role":"assistant",
                    "content":answer
                }
            )
elif page=="Download Report":
    st.header("Download AI Report")
    if "analysis_result" not in st.session_state:
        st.warning("Run AI Analysis first.")
    else:
        report=str(st.session_state["analysis_result"])
        st.download_button(
            label="Download Report",
            data=report,
            file_name="AI_Report.txt",
            mime="text/plain"
        )
        st.success("Report Ready")
