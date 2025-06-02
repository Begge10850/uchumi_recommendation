# 📦 UCHUMI STORE Recommender System

## 📝 Overview
This project implements an end-to-end retail recommender system for the “UCHUMI STORE,” guiding users to discover products that are similar (same‐category) or frequently co‐purchased. The pipeline begins with raw event and item data, proceeds through data cleaning and similarity computation, and culminates in a live Streamlit web application that fetches artifacts from AWS S3 and presents personalized recommendations. 🚀

## 📂 Project Structure
1. .gitignore
2. bundle_models.py
3. make_artifacts.py
4. requirements.txt
5. retail_cleaning.ipynb
6. uchumi.py

# .gitignore 🛡️
Specifies files and directories to exclude from version control (e.g., large pickle files, model artifacts, .streamlit/, intermediate notebooks).

# retail_cleaning.ipynb 🧹
Jupyter notebook that:

🗃️ Loads raw CSVs (events, category tree, item properties).

🔄 Cleans and filters the event dataset (timestamp conversion, null checks, selecting relevant transaction events).

📊 Builds item–item similarity (e.g., using cosine similarity on item feature vectors or co‐occurrence counts).

📦 Produces two primary artifacts:

1. df_filtered.pkl: Filtered DataFrame of transactions.

2. item_similarity.pkl: Square matrix (DataFrame) of pairwise similarity scores between items.

- 🗺️ Also outputs mapping dictionaries:

1. item_to_category.pkl: Maps each item ID to its category.

2. category_to_items.pkl: For each category, lists all item IDs belonging to it.

# make_artifacts.py ⚙️
Loads the precomputed artifacts from disk:

sim_df       = joblib.load("item_similarity.pkl")

df_filtered  = joblib.load("df_filtered.pkl")

item_to_cat  = joblib.load("item_to_category.pkl")

cat_to_items = joblib.load("category_to_items.pkl")

# 🤝 Computes “neighbors”:
For each item, retains up to 10 neighbors whose similarity ≥ 0.75 and belong to the same category. 
Stores the top‐10 sorted neighbor item IDs in a dictionary:

sim_neighbors[item_id] = [neighbor_id_1, neighbor_id_2, …]  # sorted by descending similarity

# 🔢 Computes transaction counts:
Counts how often each item appears in a “transaction” event:

counts = df_filtered[df_filtered.event == "transaction"].itemid.value_counts().to_dict()

# 💾 Stores final artifacts (via joblib.dump):
sim_neighbors.pkl
counts.pkl
Re‐saves item_to_category.pkl and category_to_items.pkl for downstream use.

# bundle_models.py 📦
-- Packages the four final artifact files (sim_neighbors.pkl, counts.pkl, item_to_category.pkl, category_to_items.pkl) into a compressed archive (models.tar.gz) so they can be uploaded/deployed as a single object.

import tarfile

files = [
    "sim_neighbors.pkl",
    "counts.pkl",
    "item_to_category.pkl",
    "category_to_items.pkl",
]

with tarfile.open("models.tar.gz", "w:gz") as tar:
    for f in files:
        tar.add(f)
print("✅ models.tar.gz created")

# requirements.txt
Lists Python dependencies:

- streamlit: Builds the web UI. 🖥️
- boto3: Interacts with AWS S3 to download models.tar.gz. ☁️
- joblib: Efficient serialization/deserialization of Python objects. 🔧
- uchumi.py 💻

# uchumi.py
- Streamlit application that:

💾 Downloads and caches model artifacts from an S3 bucket (retail-recommender/models.tar.gz) using AWS credentials configured in st.secrets.
📂 Extracts all .pkl files into a local models/ directory.
🧠 Loads them into memory as four Python objects: 

    sim_neighbors = art["sim_neighbors.pkl"]
    counts        = art["counts.pkl"]
    t2c           = art["item_to_category.pkl"]
    c2i           = art["category_to_items.pkl"]

# 🔧 Defines helper functions:
  - get_valid_items(): Returns a list of item IDs (keys of sim_neighbors).
  - fetch_recommendations(item_id, top_n=5):
    • Retrieves up to top_n “bought together” items from sim_neighbors[item_id]. 🛍️
    • Identifies “fallback” items by selecting the most popular items in the same category (excluding the selected item) from the counts dictionary, limited to 4 candidates. ⭐

# 🖼️ Builds the Streamlit UI:
  - 📄 Title and instruction markdown.
  - 🔽 Dropdown (st.selectbox) to choose any valid item ID.
  - ➕ “Add to Basket” button for the selected item.
  - 📊 Displays two sections:
    • **Items in the Same Category** (“fallback_items”):  
      - Lists each fallback item with an “Add” button.  
    • **Items Bought Together** (“bought_together”):  
      - Lists recommended co‐purchased items with “Add” buttons.

# 🛒 Sidebar Basket Panel:
  - Shows items currently in st.session_state.basket.
  - 🚮 Allows removing individual items or clearing the entire basket.

# 🚀 Installation & Setup

Clone the repository (or copy these files into a local folder).

Prepare a virtual environment (recommended):
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt

# Data Cleaning & Artifact Generation
🔔 Note: You need local copies of the raw CSVs referenced in retail_cleaning.ipynb (e.g., events.csv, category_tree.csv, item_properties.csv). Update paths inside the notebook if necessary.

1️⃣ Open and run retail_cleaning.ipynb in a Jupyter environment.

   This creates:
   1. df_filtered.pkl
   2. item_similarity.pkl
   3. item_to_category.pkl
   4. category_to_items.pkl

2️⃣ Run the artifact–building script:
   python make_artifacts.py
   Outputs:
   1. sim_neighbors.pkl
   2. counts.pkl
   3. (Re‐saves) item_to_category.pkl & category_to_items.pkl

3️⃣ Bundle all final artifacts into one archive:
   - python bundle_models.py
   - Produces models.tar.gz in the project root.

# Upload to S3
  1. Create an S3 bucket (e.g., retail-recommender) in eu-central-1 (Frankfurt). ☁️
  2. Upload models.tar.gz to the bucket under the key models.tar.gz.
  3. In your local environment, create a secrets.toml file for Streamlit (e.g., in ~/.streamlit/credentials.toml or in the repository under .streamlit/):
     [AWS]
     AWS_ACCESS_KEY_ID = "<YOUR_ACCESS_KEY_ID>"
     AWS_SECRET_ACCESS_KEY = "<YOUR_SECRET_ACCESS_KEY>"
  4. Adjust uchumi.py if your bucket name or region differs.

# Run the Streamlit App
  streamlit run uchumi.py 🏃
  This will launch a local Streamlit server (e.g., http://localhost:8501).
  The app automatically downloads models.tar.gz from S3 the first time (cached thereafter) and loads the recommendation artifacts.

# 🛠️ Tools & Libraries Used
1. Python 3.8+ 🐍

2. pandas, numpy (data wrangling in Jupyter) 📊

3. scikit-learn (e.g., cosine_similarity for computing similarity matrices) 🔬

4. joblib (serialize/deserealize large DataFrames and dictionaries) 💼

5. tarfile (bundle models into a compressed archive) 📦

6. Streamlit (front-end, interactive web UI) 🌐

7. boto3 (AWS S3 integration to fetch model artifacts) ☁️

8. AWS S3 (cloud storage for model artifacts) 📥

# 🏆 Achievements
1. End-to-End Data-Driven Recommendation Pipeline

Raw retail event data → cleaned DataFrame → item-item similarity matrix → neighbor dictionaries → deployed web app. 🎯

2. Modular Artifact Generation

Automated scripts (make_artifacts.py + bundle_models.py) allow reproducible artifact creation and bundling, decoupling computation from deployment. 🔄

3. Cloud Integration & Caching

Artifacts stored in AWS S3 and fetched on demand by Streamlit. ☁️

Caching (@st.cache_resource and @st.cache_data) ensures responsive UI and minimal repeated downloads. ⚡

4. User-Friendly UI

Intuitive dropdown to select items. 🖱️

Two recommendation strategies:

Same-category “fallback” (popular items if no strong similarity exists). 🔄

“Bought together” (co-purchase neighbors). 🛒

Session-based basket that persists selections as you navigate. 🛍️

5. Reproducibility & Extensibility

Clear project structure separates data cleaning, artifact creation, bundling, and deployment. 🔧

One can easily retrain or update similarity thresholds (e.g., change THRESHOLD in make_artifacts.py) and re-bundle for new recommendations. 🔄

# 🔐 License & Attribution
Copyright © 2025 Treva Ogwang. All Rights Reserved.
No portion of this work may be reproduced, distributed, or modified without prior written permission from the author. 🛡️

Thank you for exploring the UCHUMI STORE Recommender System! Feel free to dive into each component, tweak thresholds, or extend the UI with more advanced filtering (e.g., by price, brand, or user ratings). If you have questions or feature requests, please open an issue or pull request. ✨
