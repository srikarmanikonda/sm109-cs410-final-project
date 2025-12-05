import streamlit as st
import pandas as pd
from search_engine import ClinicalTrialsSearch


st.set_page_config(page_title="Clinical Trials Search", layout="wide")


st.title("🏥 Clinical Trials Search")
st.markdown("""
Search for clinical trials using natural language.
The system uses **BM25** for text retrieval and boosts results based on **Phase**, **Status**, and **Location** matches found in your query.
""")

@st.cache_resource
def load_engine():
    return ClinicalTrialsSearch("data/sample_trials.csv")

try:
    engine = load_engine()
    st.success("Search index loaded successfully!")
except Exception as e:
    st.error(f"Error loading search engine: {e}")
    st.stop()


query = st.text_input("Enter your search query (e.g., 'Lung cancer phase 2 recruiting in New York')", "")

if query:
    st.subheader("Search Results")


    results = engine.search(query, top_k=20)

    if results.empty:
        st.write("No results found.")
    else:

        results = results.fillna(0)


        for index, row in results.iterrows():
            score = row['final_score'] if pd.notna(row['final_score']) else 0
            with st.expander(f"{row['BriefTitle']} (Score: {score:.2f})"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**Condition:** {row['Condition']}")
                    st.markdown(f"**Summary:** {row['BriefSummary']}")

                with col2:
                    st.markdown(f"**Phase:** {row['Phase']}")
                    st.markdown(f"**Status:** {row['OverallStatus']}")
                    st.markdown(f"**Location:** {row['LocationCity']}, {row['LocationState']}")

                    st.divider()
                    bm25 = row['bm25_score'] if pd.notna(row['bm25_score']) else 0
                    boost = row['boost_score'] if pd.notna(row['boost_score']) else 0
                    st.caption(f"BM25: {bm25:.2f} | Boost: {boost:.2f}")



