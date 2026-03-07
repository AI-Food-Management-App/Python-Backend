import base64
import os
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
if not GOOGLE_VISION_API_KEY:
    raise RuntimeError("Missing GOOGLE_VISION_API_KEY in environment")

VISION_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"

VALID_INGREDIENTS = {
    "apple", "banana", "orange", "strawberry", "blueberry", "grape",
    "tomato", "cucumber", "lettuce", "carrot", "onion", "garlic",
    "pepper", "potato", "broccoli", "spinach", "mushroom",
    "chicken", "beef", "pork", "salmon", "fish", "egg", "turkey",
    "bread", "cheese", "milk", "butter", "yogurt",
    "rice", "pasta"
}

app = FastAPI(title="FoodVision Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
<<<<<<< HEAD:app.py
VISION_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"

VALID_INGREDIENTS = {
    "apple", "banana", "orange", "strawberry", "blueberry", "grape",
    "tomato", "cucumber", "lettuce", "carrot", "onion", "garlic",
    "pepper", "potato", "broccoli", "spinach", "mushroom",
    "chicken", "beef", "pork", "salmon", "fish", "egg", "turkey",
    "bread", "cheese", "milk", "butter", "yogurt",
    "rice", "pasta"
}
=======
>>>>>>> 270a89eaeea15dc88a7d881986b6b25ab766b426:ml_service/app.py

def extract_best_food_label(vision_labels):
    if not vision_labels:
        return None

    for item in vision_labels:
        label = (item.get("description") or "").lower().strip()
        if label in VALID_INGREDIENTS:
            return label.capitalize()

    return vision_labels[0].get("description")

def call_google_vision(image_bytes: bytes):
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [
            {
                "image": {"content": img_base64},
                "features": [{"type": "LABEL_DETECTION", "maxResults": 10}],
            }
        ]
    }

    resp = requests.post(VISION_URL, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data["responses"][0].get("labelAnnotations", [])

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/detect")
async def detect_food(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        labels = call_google_vision(image_bytes)
        ingredient = extract_best_food_label(labels)
        return JSONResponse({"ingredient": ingredient})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})