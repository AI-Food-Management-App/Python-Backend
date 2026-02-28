import io
import base64
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
load_dotenv()



GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
if not GOOGLE_VISION_API_KEY:
    raise RuntimeError("Missing GOOGLE_VISION_API_KEY in environment")

app = FastAPI(title="FoodVision Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
VISION_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"

VALID_INGREDIENTS = {
    "apple", "banana", "orange", "strawberry", "blueberry", "grape",
    "tomato", "cucumber", "lettuce", "carrot", "onion", "garlic",
    "pepper", "potato", "broccoli", "spinach", "mushroom",

    "chicken", "beef", "pork", "salmon", "fish", "egg", "turkey",

    "bread", "cheese", "milk", "butter", "yogurt",
    "rice", "pasta"
}

def extract_best_food_label(vision_labels):
    """Pick the most accurate food ingredient from Google Vision output."""

    if not vision_labels:
        return None

    # Try to find a specific ingredient from our list
    for item in vision_labels:
        label = item["description"].lower()
        if label in VALID_INGREDIENTS:
            return label.capitalize()

    # If nothing specific found, fallback to the first Vision label
    return vision_labels[0]["description"]

def call_google_vision(image_bytes):
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "requests": [
            {
                "image": {"content": img_base64},
                "features": [{"type": "LABEL_DETECTION", "maxResults": 10}]
            }
        ]
    }

    resp = requests.post(VISION_URL, json=payload)
    resp.raise_for_status()  # if API key invalid → 403

    data = resp.json()
    labels = data["responses"][0].get("labelAnnotations", [])

    return labels


@app.post("/detect")
async def detect_food(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        raw_labels = call_google_vision(image_bytes)
        ingredient = extract_best_food_label(raw_labels)

        return JSONResponse({"ingredient": ingredient})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
