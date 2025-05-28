import os
import tarfile
import pickle
import streamlit as st
import boto3
import pandas as pd

# ─── Configuration ───────────────────────────────────────────
BUCKET_NAME = "retail-recommender"        
MODEL_TAR_KEY = "model.tar.gz"           # path in S3
REGION = "eu-central-1"                   # your AWS region

# ─── Download & load model artifacts ─────────────────────────
@st.cache_resource
def load_models():
    # Download the tar.gz if not present
    local_tar = "model.tar.gz"
    if not os.path.exists(local_tar):
        s3 = boto3.client(
            "s3",
            region_name=REGION,
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        )
        s3.download_file(BUCKET_NAME, MODEL_TAR_KEY, local_tar)

    
    extract_dir = "models"
    if not os.path.isdir(extract_dir):
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(extract_dir)

    # Load pickles from extracted folder
    artifacts = {}
    for fname in os.listdir(extract_dir):
        if fname.endswith(".pkl") or fname.endswith(".pickle"):
            path = os.path.join(extract_dir, fname)
            with open(path, "rb") as f:
                artifacts[fname] = pickle.load(f)
    return artifacts

models = load_models()
# Unpack artifacts
sim_df    = models.get("item_similarity.pkl")
t2c       = models.get("item_to_category.pkl")
c2i       = models.get("category_to_items.pkl")
df_filtered = models.get("df_filtered.pkl")

# ─── Helper functions ──────────────────────────────────────────
def get_valid_items():
    return list(sim_df.index)

@st.cache_data(ttl=300)
def fetch_recommendations(item_id: int, top_n: int = 5, threshold: float = 0.75):
    # If item not in index
    if item_id not in sim_df.index:
        return {"error": f"Item ID {item_id} not found."}

    # Bought together
    s = sim_df.loc[item_id]
    s = s[(s >= threshold) & (s.index != item_id)]
    cat = t2c.get(item_id)
    if cat:
        s = s[s.index.map(t2c.get) == cat]
    bought = s.sort_values(ascending=False).head(top_n).index.tolist()

    # Fallback: popular in same category
    fallback = []
    if cat:
        candidates = [i for i in c2i.get(cat, []) if i != item_id and i in sim_df.index]
        if candidates:
            counts = df_filtered[df_filtered.event == "transaction"].itemid.value_counts()
            fallback = sorted(
                candidates,
                key=lambda i: counts.get(i, 0),
                reverse=True
            )[:4]
    return {"bought_together": bought, "fallback_items": fallback}

# ─── Streamlit UI ─────────────────────────────────────────────
st.title("🍭 UCHUMI STORE")
st.markdown("Please click the dropdown menu to access the products.")

valid_items = get_valid_items()
selected_item = st.selectbox("", ["Select an item..."] + valid_items)

# Initialize basket
if "basket" not in st.session_state:
    st.session_state.basket = []

if selected_item != "Select an item...":
    st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
    st.markdown(f"### 🔎 You selected: Item {selected_item}")
    if st.button("Add Selected Item to Basket"):
        if selected_item not in st.session_state.basket:
            st.session_state.basket.append(selected_item)

    result = fetch_recommendations(int(selected_item))
    fallback_items = result.get("fallback_items", [])
    bought_together = result.get("bought_together", [])

    st.markdown("---")
    st.markdown("### 🧩 Similar items in the same category")
    if fallback_items:
        for item in fallback_items:
            col1, col2 = st.columns([4,1])
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
            col1, col2 = st.columns([4,1])
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
        col1, col2 = st.sidebar.columns([4,1])
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


'''''
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
'''
