import streamlit as st
import boto3
import json
import os

# ─── Configuration ───
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "retail-recommender-endpoint")
REGION = os.getenv("AWS_REGION", "eu-central-1")

# ─── Initialize SageMaker runtime client ───
@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

runtime = get_runtime_client()

# ─── Fetch valid product IDs from endpoint ───
@st.cache_data(ttl=600)
def get_valid_items():
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"get_index": True})
    )
    return json.loads(response["Body"].read().decode()).get("valid_items", [])

# ─── Call endpoint to fetch recommendations ───
@st.cache_data(ttl=300, show_spinner=False)
def fetch_recommendations(item_id: int):
    payload = {"item_id": item_id, "top_n": 5, "threshold": 0.75}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    return json.loads(response["Body"].read().decode())

# ─── UI ───
st.title("🍭 UCHUMI STORE")
st.markdown("Please click the dropdown menu to access the products.")

valid_dropdown_items = get_valid_items()

selected_item = st.selectbox("", ["Select an item..."] + valid_dropdown_items)

# Initialize basket in session state
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

    # Get predictions from SageMaker
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
