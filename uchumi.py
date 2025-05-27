import streamlit as st
import boto3
import json
import os
import joblib
import pandas as pd

# ─── Configuration ───
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "retail-recommender-endpoint")
REGION = os.getenv("AWS_REGION", "eu-central-1")

# ─── Load item index for dropdown ───
try:
    item_similarity_df = joblib.load("item_similarity.pkl")
    valid_dropdown_items = list(item_similarity_df.index.astype(str))
except:
    valid_dropdown_items = []

# ─── Initialize SageMaker runtime client ───
@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

runtime = get_runtime_client()

# ─── Call endpoint to fetch predictions ───
@st.cache_data(ttl=300, show_spinner=False)
def fetch_recommendations(item_id: int):
    payload = {"item_id": item_id, "top_n": 5, "threshold": 0.75}  # internal control
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    result = json.loads(response["Body"].read().decode())
    return result

# ─── UI ───
st.title("🍭 UCHUMI STORE")
st.markdown("Please click the dropdown menu to access the products.")

selected_item = st.selectbox("", ["Select an item..."] + valid_dropdown_items)

if 'basket' not in st.session_state:
    st.session_state.basket = []

if selected_item != "Select an item...":
    st.markdown("""
        <div style="margin-top: 30px;"></div>
    """, unsafe_allow_html=True)

    st.markdown(f"### 🔎 You selected: Item {selected_item}")
    if st.button("Add Selected Item to Basket"):
        if selected_item not in st.session_state.basket:
            st.session_state.basket.append(selected_item)

    # Call SageMaker endpoint to get recommendations
    result = fetch_recommendations(int(selected_item))
    fallback_items = result.get("fallback_items", [])
    bought_together = result.get("bought_together", [])

    st.markdown("---")
    st.markdown("### 🧩 Similar items in the same category")
    if fallback_items:
        for item in fallback_items:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"- Item {item}")
            with col2:
                if st.button("Add", key=f"add_fallback_{item}"):
                    if item not in st.session_state.basket:
                        st.session_state.basket.append(item)
    else:
        st.info("No similar items found in the same category.")

    st.markdown("---")
    st.markdown("### 💹 Items bought together")
    if bought_together:
        for item in bought_together:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"- Item {item}")
            with col2:
                if st.button("Add", key=f"add_bought_{item}"):
                    if item not in st.session_state.basket:
                        st.session_state.basket.append(item)
    else:
        st.info("No co-purchased items found.")

# Sidebar basket panel
st.sidebar.title("🧺 Your Basket")
if st.session_state.basket:
    for i, item in enumerate(st.session_state.basket):
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.sidebar.write(f"Item {item}")
        with col2:
            if st.sidebar.button("❌", key=f"remove_{i}"):
                st.session_state.basket.pop(i)
                st.experimental_rerun()
    if st.sidebar.button("Clear Basket"):
        st.session_state.basket.clear()
        st.experimental_rerun()
else:
    st.sidebar.write("Your basket is empty.")
