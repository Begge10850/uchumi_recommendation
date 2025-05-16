import os
import boto3
import joblib
import pandas as pd

# Define bucket and model file names
BUCKET_NAME = "retail-recommender-treva"
MODEL_FILES = {
    "item_similarity": "models/item_similarity.pkl",
    "item_to_category": "models/item_to_category.pkl",
    "category_to_items": "models/category_to_items.pkl",
    "df_filtered": "models/df_filtered.pkl"
}

# Local path to temporarily store models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_from_s3():
    print("🔁 Downloading model files from S3...")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "eu-north-1")
    )

    for key, filename in MODEL_FILES.items():
        local_path = os.path.join(MODEL_DIR, filename)
        print(f"⬇️ Downloading {filename}...")
        s3.download_file(BUCKET_NAME, filename, local_path)

    print("✅ All models downloaded.")

def load_model():
    download_from_s3()

    print("📦 Loading model files into memory...")
    item_similarity = joblib.load(os.path.join(MODEL_DIR, MODEL_FILES["item_similarity"]))
    item_to_category = joblib.load(os.path.join(MODEL_DIR, MODEL_FILES["item_to_category"]))
    category_to_items = joblib.load(os.path.join(MODEL_DIR, MODEL_FILES["category_to_items"]))
    df_filtered = joblib.load(os.path.join(MODEL_DIR, MODEL_FILES["df_filtered"]))

    return {
        "item_similarity": item_similarity,
        "item_to_category": item_to_category,
        "category_to_items": category_to_items,
        "df_filtered": df_filtered
    }

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

    # Cosine similarity recommendations
    if item_id in item_similarity_df.index:
        similar_items = item_similarity_df.loc[item_id]
        filtered_similar = similar_items[(similar_items >= threshold) & (similar_items.index != item_id)]
        if item_cat:
            filtered_similar = filtered_similar[filtered_similar.index.map(item_to_category.get) == item_cat]
        bought_together = filtered_similar.sort_values(ascending=False).head(top_n).index.tolist()

    # Same category fallback
    if item_cat:
        cat_items = category_to_items.get(item_cat, [])
        valid_cat_items = [item for item in cat_items if item != item_id and item in item_similarity_df.index]
        if valid_cat_items:
            purchase_counts = df_filtered[df_filtered['event'] == 'transaction']['itemid'].value_counts()
            sorted_cat_items = sorted(valid_cat_items, key=lambda x: purchase_counts.get(x, 0), reverse=True)
            fallback_items = sorted_cat_items[:4]

    return {
        "selected_item": item_id,
        "items_bought_together": bought_together,
        "similar_items_in_category": fallback_items
    }
