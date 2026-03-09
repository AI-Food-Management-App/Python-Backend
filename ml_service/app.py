import io
import base64
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import re
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
    # Fruits
    "apple", "banana", "orange", "strawberry", "blueberry", "grape",
    "mango", "pineapple", "watermelon", "lemon", "lime", "peach",
    "pear", "cherry", "raspberry", "blackberry", "kiwi", "melon",
    "avocado", "coconut", "fig", "plum", "apricot", "grapefruit", "pomegranate", 
    "papaya", "passionfruit", "lychee", "guava", "dragonfruit", "durian", "jackfruit", "starfruit", 
    "persimmon", "cantaloupe", "honeydew", "cranberry", "currant", "elderberry", "gooseberry", 
    "mulberry", "olive", "date", "prune", "raisin", "sultana", "currant", "clementine", 
    "tangerine", "mandarin", "blood orange", "kumquat", "yuzu", "satsuma", "ugli fruit", 
    "calamansi", "bergamot", "finger lime", 

    # Vegetables
    "tomato", "cucumber", "lettuce", "carrot", "onion", "garlic",
    "pepper", "potato", "broccoli", "spinach", "mushroom", "celery",
    "cauliflower", "cabbage", "zucchini", "pumpkin", "beetroot",
    "sweetcorn", "corn", "peas", "beans", "lentils", "chickpeas",
    "asparagus", "artichoke", "leek", "radish", "turnip", "parsnip", 
    "brussels sprouts", "kale", "collard greens", "arugula", 
    "watercress", "bok choy", "endive", "fennel", "okra", "squash", 
    "butternut squash", "acorn squash", "spaghetti squash", "zucchini", 
    "yellow squash", 

    # Meat & Fish
    "chicken", "beef", "pork", "salmon", "fish", "egg", "turkey",
    "lamb", "tuna", "shrimp", "prawn", "cod", "bacon", "ham",
    "sausage", "mince", "duck", "goose", "venison", "rabbit", 
    "crab", "lobster", "mussels", "oysters", "scallops", "squid", 
    "octopus", "calamari", "caviar", "roe", "anchovy", "sardine", 
    "herring", "trout", "bass", "snapper", "tilapia", "halibut", 
    "flounder", "sole", "mahi mahi", "grouper", "catfish", "barramundi", 
    "red snapper", "sea bass", "rockfish", "cod", "pollock", "whiting", 
    "haddock", "swordfish", "sturgeon", "mackerel", "bluefish", "albacore", "yellowfin tuna",
     "bigeye tuna", "skipjack tuna", 

    # Dairy
    "cheese", "milk", "butter", "yogurt", "cream", "icecream",
    "ice cream", "cheddar", "mozzarella", "parmesan", "feta", 
    "gouda", "brie", "blue cheese", "cottage cheese", "ricotta", 
    "sour cream", "whipped cream", "custard", "evaporated milk", 
    "condensed milk", "cream cheese", "mascarpone", "clotted cream", 
    "double cream", "buttermilk", "yogurt drink", "kefir", 
    "lactose free milk", "almond milk", "soy milk", "oat milk", 
     "coconut milk", "rice milk", "hemp milk", "cashew milk", 
     "macadamia milk", "hazelnut milk", "flax milk", "pea milk", 
     "walnut milk", "quinoa milk", "sesame milk", "sunflower seed milk",
     

    # Grains & Carbs
    "bread", "rice", "pasta", "noodle", "oats", "cereal",
    "flour", "tortilla", "cracker", "biscuit", "bagel", 
    "croissant", "muffin", "pancake", "waffle", "quinoa", 
    "couscous", "bulgur", "barley", "rye", "cornbread", 
    "sourdough", "ciabatta", "focaccia", "naan", "pita", "flatbread",
    "lasagna", "spaghetti", "macaroni", "penne", "fusilli", "linguine",
    "farfalle", "ravioli", "tortellini", "gnocchi", "cannelloni", "ziti",
    "baguette", "brioche", "whole wheat bread", "multigrain bread", 
    "gluten free bread", "crispbread", "rice cake", "corn tortilla", "crisps", 
     "chips", "popcorn", "pretzel", "breadstick", "pita bread", "flatbread", "wrap", 
    "sandwich bread", "burger bun", "hot dog bun", "english muffin",

    # Pantry
    "chocolate", "sugar", "salt", "oil", "vinegar", "sauce",
    "ketchup", "mayonnaise", "mustard", "honey", "jam",
    "peanut butter", "soup", "bean", "baked beans", "canned beans", 
    "canned tomatoes", "canned corn", "canned tuna", "canned salmon", "canned chicken", "canned vegetables",
    "canned fruit", "canned soup", "canned chili", "canned stew", "canned broth", "canned stock", "canned curry", "canned pasta", "canned beans",
    "canned lentils", "canned chickpeas", "canned black beans", 
    "canned kidney beans", "canned pinto beans", "canned navy beans", 
    "canned cannellini beans",

    # Drinks
    "juice", "water", "coffee", "tea", "milk", "soda", 
    "soft drink", "wine", "beer", "cocktail", "smoothie", "milkshake", 
    "lemonade", "iced tea", "hot chocolate", "energy drink", "sports drink", "sparkling water", "mineral water", 
     "flavored water", "fruit juice", "vegetable juice", "protein shake", 
     "kombucha", "matcha", "chai tea", "herbal tea", "green tea", 
     "black tea", "white tea", "oolong tea", "pu-erh tea"
}

SKIP_WORDS = {
    "food", "ingredient", "product", "item", "meal", "dish",
    "snack", "cuisine", "recipe", "grocery", "store", "original",
    "natural", "organic", "fresh", "made", "with", "style",
    "flavour", "flavor", "brand", "package", "serving"
}

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