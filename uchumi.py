import streamlit as st
import boto3
import json
import os

# Configuration
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "retail-recommender-endpoint")
REGION = os.getenv("AWS_REGION", "eu-central-1")

@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

runtime = get_runtime_client()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_recommendations(item_id: int, top_n: int = 5, threshold: float = 0.75):
    payload = {"item_id": item_id, "top_n": top_n, "threshold": threshold}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    result = json.loads(response["Body"].read().decode())
    return result

st.title("🛍️ Product Recommendation System")
st.markdown("Please select a product to get recommendations.")

# Example list of valid items - ideally fetch or cache from a config or API
# Here we assume item IDs are integers
item_input = st.number_input("Enter Item ID", min_value=1, step=1)
top_n = st.slider("How many suggestions?", 1, 20, 5)
threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.75, 0.01)

if st.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        result = fetch_recommendations(int(item_input), top_n, threshold)
    if "error" in result:
        st.error(result["error"])
    else:
        bought = result.get("bought_together", [])
        fallback = result.get("fallback_items", [])

        st.subheader("🛒 Bought Together")
        if bought:
            for i, it in enumerate(bought):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"- Item {it}")
                with col2:
                    if st.button("Add", key=f"add_bought_{it}"):
                        st.session_state.basket.append(it)
        else:
            st.info("No co-purchased items found.")

        st.subheader("🧩 Similar Items in Same Category")
        if fallback:
            for it in fallback:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"- Item {it}")
                with col2:
                    if st.button("Add", key=f"add_fallback_{it}"):
                        st.session_state.basket.append(it)
        else:
            st.info("No similar items found in category.")

# Sidebar for basket
if 'basket' not in st.session_state:
    st.session_state.basket = []
st.sidebar.title("🧺 Your Basket")
if st.session_state.basket:
    for idx, it in enumerate(st.session_state.basket):
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.sidebar.write(f"Item {it}")
        with col2:
            if st.sidebar.button("❌", key=f"remove_{idx}"):
                st.session_state.basket.pop(idx)
                st.experimental_rerun()
    if st.sidebar.button("Clear Basket"):
        st.session_state.basket.clear()
        st.experimental_rerun()
else:
    st.sidebar.write("Your basket is empty.")
