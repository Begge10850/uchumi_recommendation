import os
import tarfile
import pickle
import streamlit as st
import boto3

# ─── Configuration ───────────────────────────────────────────
BUCKET_NAME = "retail-recommender"
MODEL_TAR_KEY = "models.tar.gz"
REGION = "eu-central-1"

# ─── Download & load model artifacts ─────────────────────────
@st.cache_resource
def load_artifacts():
    local_tar = MODEL_TAR_KEY
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
    artifacts = {}
    for fname in os.listdir(extract_dir):
        if fname.endswith(".pkl") or fname.endswith(".pickle"):
            path = os.path.join(extract_dir, fname)
            with open(path, "rb") as f:
                artifacts[fname] = pickle.load(f)
    return artifacts

# ─── Load artifacts ───────────────────────────────────────────
art = load_artifacts()
sim_neighbors = art.get("sim_neighbors.pkl", {})
counts        = art.get("counts.pkl", {})
t2c           = art.get("item_to_category.pkl", {})
c2i           = art.get("category_to_items.pkl", {})

# ─── Helper functions ─────────────────────────────────────────
def get_valid_items():
    return list(sim_neighbors.keys())

@st.cache_data(ttl=300)
def fetch_recommendations(item_id: int, top_n: int = 5):
    bought = sim_neighbors.get(item_id, [])[:top_n]
    cat = t2c.get(item_id)
    candidates = [i for i in c2i.get(cat, []) if i != item_id]
    fallback = sorted(candidates, key=lambda i: counts.get(i, 0), reverse=True)[:4]
    return {"bought_together": bought, "fallback_items": fallback}

# ─── Streamlit UI ─────────────────────────────────────────────
st.title("🍭 UCHUMI STORE")
st.markdown("Please click the dropdown menu to access the products.")

valid_items = get_valid_items()
selected_item = st.selectbox("", ["Select an item..."] + valid_items)

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

# ─── Sidebar basket panel ─────────────────────────────────────
def remove_item(item_id):
    st.session_state.basket = [x for x in st.session_state.basket if x != item_id]

def clear_basket():
    st.session_state.basket.clear()

st.sidebar.title("🧺 Your Basket")
if st.session_state.basket:
    for item in st.session_state.basket:
        cols = st.sidebar.columns([4, 1])
        cols[0].write(f"Item {item}")
        cols[1].button("❌", key=f"remove_{item}", on_click=remove_item, args=(item,))
    st.sidebar.button("Clear Basket", on_click=clear_basket)
else:
    st.sidebar.write("Your basket is empty.")
