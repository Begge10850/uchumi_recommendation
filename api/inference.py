import joblib
import pandas as pd
import os
import gdown

# File IDs extracted from your shared links
FILE_IDS = {
    "item_similarity": "1OIaMWUWdERg--Q_P8DyoVR2LgY_o19ck",
    "item_to_category": "1KqxM3iYxJ80qOarnIyexjzyEqfFcibV_",
    "category_to_items": "1N_5kOiivNwKH4ncv08oAej8BdRuYGHCZ",
    "df_filtered": "1y5giweAx8N85aTA2vudQronPsYKjKHYU"
}

# Download and load models
def load_model():
    os.makedirs("models", exist_ok=True)
    models = {}

    for name, file_id in FILE_IDS.items():
        output_path = f"models/{name}.pkl"
        if not os.path.exists(output_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"📦 Downloading {name}.pkl...")
            gdown.download(url, output_path, quiet=False)
        models[name] = joblib.load(output_path)

    print("✅ All model files loaded.")
    return models

# Prediction logic
def predict_fn(input_data, model):
    item_id = input_data.get("item_id")
    top_n = input_data.get("top_n", 5)
    threshold = input_data.get("threshold", 0.75)

    item_similarity_df = model["item_similarity"]
    item_to_category = model["item_to_category"]
    category_to_items = model["category_to_items"]
    df_filtered = model["df_filtered"]

    item_cat = item_to_category.get(item_id)
    bought_together = []
    fallback_items = []

    # Similar items by cosine similarity
    if item_id in item_similarity_df.index:
        similar_items = item_similarity_df.loc[item_id]
        filtered = similar_items[(similar_items >= threshold) & (similar_items.index != item_id)]
        if item_cat:
            filtered = filtered[filtered.index.map(item_to_category.get) == item_cat]
        bought_together = filtered.sort_values(ascending=False).head(top_n).index.tolist()

    # Top purchased items in same category
    if item_cat:
        cat_items = category_to_items.get(item_cat, [])
        valid_items = [item for item in cat_items if item != item_id and item in item_similarity_df.index]
        if valid_items:
            purchase_counts = df_filtered[df_filtered["event"] == "transaction"]["itemid"].value_counts()
            fallback_items = sorted(valid_items, key=lambda x: purchase_counts.get(x, 0), reverse=True)[:4]

    return {
        "selected_item": item_id,
        "items_bought_together": bought_together,
        "similar_items_in_category": fallback_items
    }
