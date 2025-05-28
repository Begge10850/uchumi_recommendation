# bundle_models.py
import tarfile

files = [
    "sim_neighbors.pkl",
    "counts.pkl",
    "item_to_category.pkl",
    "category_to_items.pkl",
]

with tarfile.open("models.tar.gz", "w:gz") as tar:
    for f in files:
        print(f"Adding {f}…")
        tar.add(f)
print("✅ models.tar.gz created")