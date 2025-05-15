import joblib
import pandas as pd

# Load model files once at app startup
def load_model():
    print("📦 Loading model files from 'models/'...")

    item_similarity = joblib.load("models/item_similarity.pkl")
    item_to_category = joblib.load("models/item_to_category.pkl")
    category_to_items = joblib.load("models/category_to_items.pkl")
    df_filtered = joblib.load("models/df_filtered.pkl")

    print("✅ Files loaded successfully.")

    return {
        "item_similarity": item_similarity,
        "item_to_category": item_to_category,
        "category_to_items": category_to_items,
        "df_filtered": df_filtered
    }

# Prediction logic
def predict_fn(input_data, model):
    # Convert to float since your index is float64
    item_id = float(input_data.get("item_id"))
    top_n = input_data.get("top_n", 5)
    threshold = input_data.get("threshold", 0.75)

    item_similarity_df = model["item_similarity"]
    item_to_category = model["item_to_category"]
    category_to_items = model["category_to_items"]
    df_filtered = model["df_filtered"]

    item_cat = item_to_category.get(item_id)
    bought_together = []
    fallback_items = []

    # Get similar items using cosine similarity
    if item_id in item_similarity_df.index:
        similar_items = item_similarity_df.loc[item_id]
        filtered_similar = similar_items[(similar_items >= threshold) & (similar_items.index != item_id)]

        if item_cat:
            filtered_similar = filtered_similar[
                filtered_similar.index.map(item_to_category.get) == item_cat
            ]
        bought_together = filtered_similar.sort_values(ascending=False).head(top_n).index.tolist()

    # Fallback: top purchased items in the same category
    if item_cat:
        cat_items = category_to_items.get(item_cat, [])
        valid_cat_items = [item for item in cat_items if item != item_id and item in item_similarity_df.index]
        if valid_cat_items:
            purchase_counts = df_filtered[df_filtered["event"] == "transaction"]["itemid"].value_counts()
            sorted_cat_items = sorted(valid_cat_items, key=lambda x: purchase_counts.get(x, 0), reverse=True)
            fallback_items = sorted_cat_items[:4]

    return {
        "selected_item": item_id,
        "items_bought_together": bought_together,
        "similar_items_in_category": fallback_items
    }
