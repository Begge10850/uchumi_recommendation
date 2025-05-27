# ─── 1) logging + standard imports ─────────────────────────────────────────────
import logging, os, json, joblib, pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Pandas version: %s", pd.__version__)

# ─── 2) model_fn: load your pickles ─────────────────────────────────────────────
def model_fn(model_dir):
    logger.info("Loading model artifacts from %s", model_dir)
    try:
        sim   = joblib.load(os.path.join(model_dir, "item_similarity.pkl"))
        t2c   = joblib.load(os.path.join(model_dir, "item_to_category.pkl"))
        c2i   = joblib.load(os.path.join(model_dir, "category_to_items.pkl"))
        df    = joblib.load(os.path.join(model_dir, "df_filtered.pkl"))
        logger.info("Artifacts loaded successfully")
        return {
            "item_similarity":    sim,
            "item_to_category":   t2c,
            "category_to_items":  c2i,
            "df_filtered":        df
        }
    except Exception as e:
        logger.error("Failed to load model files: %s", e)
        raise

# ─── 3) input_fn ────────────────────────────────────────────────────────────────
def input_fn(request_body, content_type='application/json'):
    if content_type != 'application/json':
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    logger.info("Received payload: %s", payload)
    return payload

# ─── 4) predict_fn ─────────────────────────────────────────────────────────────
def predict_fn(input_data, model):
    if input_data.get("get_index"):
        return {"valid_items": list(model["item_similarity"].index)}

    if "item_id" not in input_data:
        raise ValueError("Missing required field: item_id")

    item_id   = input_data["item_id"]
    top_n     = input_data.get("top_n", 5)
    threshold = input_data.get("threshold", 0.75)

    sim_df  = model["item_similarity"]
    to_cat  = model["item_to_category"]
    cat2its = model["category_to_items"]
    df      = model["df_filtered"]

    if item_id not in sim_df.index:
        return { "error": f"Item ID {item_id} not in similarity matrix" }

    # co-purchased
    s = sim_df.loc[item_id]
    s = s[(s >= threshold) & (s.index != item_id)]
    cat = to_cat.get(item_id)
    if cat:
        s = s[s.index.map(to_cat.get) == cat]
    bought = s.sort_values(ascending=False).head(top_n).index.tolist()

    # fallback: popular in same category
    fallback = []
    if cat:
        candidates = [i for i in cat2its.get(cat, []) if i != item_id and i in sim_df.index]
        if candidates:
            counts = df[df.event=="transaction"].itemid.value_counts()
            fallback = sorted(
                candidates,
                key=lambda i: counts.get(i, 0),
                reverse=True
            )[:4]

    return {"bought_together": bought, "fallback_items": fallback}

# ─── 5) output_fn ──────────────────────────────────────────────────────────────
def output_fn(prediction, content_type='application/json'):
    if content_type != 'application/json':
        raise ValueError(f"Unsupported content type: {content_type}")
    return json.dumps(prediction)
