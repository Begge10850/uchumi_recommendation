from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.inference import load_model, predict_fn  # ✅ Centralized logic

# 1. Create FastAPI app instance
app = FastAPI()

# 2. Load model once on startup
model = load_model()

# 3. Define input format
class ItemRequest(BaseModel):
    item_id: float  # ✅ match the model's dtype (float64)
    top_n: int = 5
    threshold: float = 0.75

# 4. Define the recommendation endpoint
@app.post("/recommend")
def get_recommendations(request: ItemRequest):
    item_id = request.item_id

    # ✅ Correct dtype-safe check
    if item_id not in model["item_similarity"].index:
        raise HTTPException(status_code=404, detail="Item not found in model")

    response = predict_fn(
        input_data={
            "item_id": item_id,
            "top_n": request.top_n,
            "threshold": request.threshold
        },
        model=model
    )

    return response
