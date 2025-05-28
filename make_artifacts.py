import joblib

# 1) Load your old artifacts (these use pandas, run in your pandas env)
sim_df = joblib.load("item_similarity.pkl")
df     = joblib.load("df_filtered.pkl")
t2c    = joblib.load("item_to_category.pkl")
c2i    = joblib.load("category_to_items.pkl")

# 2) Precompute for each item the sorted list of neighbours ≥ threshold
THRESHOLD = 0.75
MAX_NEI   = 10  # store up to 10 neighbors (you'll slice to top_n=5 at runtime)
sim_neighbors = {}
for item in sim_df.index:
    row = sim_df.loc[item]
    # filter & category-match
    neigh = row[(row >= THRESHOLD) & (row.index != item)]
    cat  = t2c.get(item)
    if cat is not None:
        neigh = neigh[neigh.index.map(lambda i: t2c.get(i) == cat)]
    # sort descending, take top MAX_NEI
    sim_neighbors[item] = neigh.sort_values(ascending=False).index.tolist()[:MAX_NEI]

# 3) Build your transaction counts dict
counts = df[df.event == "transaction"].itemid.value_counts().to_dict()

# 4) Dump everything as pure Python via joblib
joblib.dump(sim_neighbors,       "sim_neighbors.pkl")
joblib.dump(counts,             "counts.pkl")
joblib.dump(t2c,                "item_to_category.pkl")
joblib.dump(c2i,                "category_to_items.pkl")