import streamlit as st
import joblib
import pandas as pd

# Load saved model files
item_similarity_df = joblib.load("item_similarity.pkl")
item_to_category = joblib.load("item_to_category.pkl")
category_to_items = joblib.load("category_to_items.pkl")
df_filtered = joblib.load("df_filtered.pkl")  # Your cleaned event data with transactions

# Helper function
def recommend_items(item_id, top_n=5, threshold=0.75):
    item_cat = item_to_category.get(item_id)
    bought_together = []
    fallback_items = []

    # Step 1: Cosine similarity (items bought together)
    if item_id in item_similarity_df.index:
        similar_items = item_similarity_df.loc[item_id]
        filtered_similar = similar_items[(similar_items >= threshold) & (similar_items.index != item_id)]

        if item_cat:
            filtered_similar = filtered_similar[
                filtered_similar.index.map(item_to_category.get) == item_cat
            ]

        bought_together = filtered_similar.sort_values(ascending=False).head(top_n).index.tolist()

    # Step 2: Most purchased items in same category (fallback)
    if item_cat:
        cat_items = category_to_items.get(item_cat, [])
        valid_cat_items = [item for item in cat_items if item != item_id and item in item_similarity_df.index]

        if valid_cat_items:
            purchase_counts = df_filtered[df_filtered['event'] == 'transaction']['itemid'].value_counts()
            sorted_cat_items = sorted(valid_cat_items, key=lambda x: purchase_counts.get(x, 0), reverse=True)
            fallback_items = sorted_cat_items[:4]

    return fallback_items, bought_together

# Filter dropdown items to only those that exist in similarity matrix
valid_dropdown_items = list(item_similarity_df.index)

# Initialize basket in session state
if 'basket' not in st.session_state:
    st.session_state.basket = []

# Streamlit UI
st.title("🛍️ Product Recommendation System")
st.markdown("Please click the dropdown menu to access the products.")

selected_item = st.selectbox("", ["Select an item..."] + valid_dropdown_items)

if selected_item != "Select an item...":
    st.markdown("""
        <div style="margin-top: 30px;"></div>
    """, unsafe_allow_html=True)

    st.markdown(f"### 🔎 You selected: Item {selected_item}")
    if st.button("Add Selected Item to Basket"):
        if selected_item not in st.session_state.basket:
            st.session_state.basket.append(selected_item)

    fallback_items, bought_together = recommend_items(selected_item)

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
    st.markdown("### 🛍️ Items bought together")
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
        st.rerun()
else:
    st.sidebar.write("Your basket is empty.")
