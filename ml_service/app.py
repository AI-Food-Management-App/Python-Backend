"""FoodVision ML Service - FastAPI application for food detection."""
import base64
import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    "papaya", "passionfruit", "lychee", "guava", "dragonfruit", "durian",
    "jackfruit", "starfruit", "persimmon", "cantaloupe", "honeydew",
    "cranberry", "currant", "elderberry", "gooseberry", "mulberry", "olive",
    "date", "prune", "raisin", "sultana", "clementine", "tangerine", "mandarin",
    "blood orange", "kumquat", "yuzu", "satsuma", "ugli fruit", "calamansi",
    "bergamot", "finger lime",

    # Vegetables
    "tomato", "cucumber", "lettuce", "carrot", "onion", "garlic",
    "pepper", "potato", "broccoli", "spinach", "mushroom", "celery",
    "cauliflower", "cabbage", "zucchini", "pumpkin", "beetroot",
    "sweetcorn", "corn", "peas", "beans", "lentils", "chickpeas",
    "asparagus", "artichoke", "leek", "radish", "turnip", "parsnip",
    "brussels sprouts", "kale", "collard greens", "arugula", "watercress",
    "bok choy", "endive", "fennel", "okra", "squash", "butternut squash",
    "acorn squash", "spaghetti squash", "yellow squash",

    # Meat & Fish
    "chicken", "beef", "pork", "salmon", "fish", "egg", "turkey",
    "lamb", "tuna", "shrimp", "prawn", "cod", "bacon", "ham",
    "sausage", "mince", "duck", "goose", "venison", "rabbit", "crab",
    "lobster", "mussels", "oysters", "scallops", "squid", "octopus",
    "calamari", "caviar", "roe", "anchovy", "sardine", "herring", "trout",
    "bass", "snapper", "tilapia", "halibut", "flounder", "sole", "mahi mahi",
    "grouper", "catfish", "barramundi", "red snapper", "sea bass", "rockfish",
    "pollock", "whiting", "haddock", "swordfish", "sturgeon", "mackerel",
    "bluefish", "albacore", "yellowfin tuna", "bigeye tuna", "skipjack tuna",

    # Dairy
    "cheese", "milk", "butter", "yogurt", "cream", "icecream", "ice cream",
    "cheddar", "mozzarella", "parmesan", "feta", "gouda", "brie",
    "blue cheese", "cottage cheese", "ricotta", "sour cream", "whipped cream",
    "custard", "evaporated milk", "condensed milk", "cream cheese", "mascarpone",
    "clotted cream", "double cream", "buttermilk", "yogurt drink", "kefir",
    "lactose free milk", "almond milk", "soy milk", "oat milk", "coconut milk",
    "rice milk", "hemp milk", "cashew milk", "macadamia milk", "hazelnut milk",
    "flax milk", "pea milk", "walnut milk", "quinoa milk", "sesame milk",
    "sunflower seed milk",

    # Grains & Carbs
    "bread", "rice", "pasta", "noodle", "oats", "cereal", "flour", "tortilla",
    "cracker", "biscuit", "bagel", "croissant", "muffin", "pancake", "waffle",
    "quinoa", "couscous", "bulgur", "barley", "rye", "cornbread", "sourdough",
    "ciabatta", "focaccia", "naan", "pita", "flatbread", "lasagna", "spaghetti",
    "macaroni", "penne", "fusilli", "linguine", "farfalle", "ravioli",
    "tortellini", "gnocchi", "cannelloni", "ziti", "baguette", "brioche",
    "whole wheat bread", "multigrain bread", "gluten free bread", "crispbread",
    "rice cake", "corn tortilla", "crisps", "chips", "popcorn", "pretzel",
    "breadstick", "pita bread", "wrap", "sandwich bread", "burger bun",
    "hot dog bun", "english muffin",

    # Pantry
    "chocolate", "sugar", "salt", "oil", "vinegar", "sauce", "ketchup",
    "mayonnaise", "mustard", "honey", "jam", "peanut butter", "soup", "bean",
    "baked beans", "canned beans", "canned tomatoes", "canned corn",
    "canned tuna", "canned salmon", "canned chicken", "canned vegetables",
    "canned fruit", "canned soup", "canned chili", "canned stew", "canned broth",
    "canned stock", "canned curry", "canned pasta", "canned lentils",
    "canned chickpeas", "canned black beans", "canned kidney beans",
    "canned pinto beans", "canned navy beans", "canned cannellini beans",

    # Drinks
    "juice", "water", "coffee", "tea", "soda", "soft drink", "wine", "beer",
    "cocktail", "smoothie", "milkshake", "lemonade", "iced tea", "hot chocolate",
    "energy drink", "sports drink", "sparkling water", "mineral water",
    "flavored water", "fruit juice", "vegetable juice", "protein shake",
    "kombucha", "matcha", "chai tea", "herbal tea", "green tea", "black tea",
    "white tea", "oolong tea", "pu-erh tea"
}

SKIP_WORDS = {
    "food", "ingredient", "product", "item", "meal", "dish",
    "snack", "cuisine", "recipe", "grocery", "store", "original",
    "natural", "organic", "fresh", "made", "with", "style",
    "flavour", "flavor", "brand", "package", "serving"
}


def call_google_vision(image_bytes):
   
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "requests": [
            {
                "image": {"content": img_base64},
                "features": [
                    {"type": "LABEL_DETECTION",      "maxResults": 15},
                    {"type": "OBJECT_LOCALIZATION",  "maxResults": 10},
                    {"type": "TEXT_DETECTION",       "maxResults": 1},
                ]
            }
        ]
    }

    resp = requests.post(VISION_URL, json=payload)
    resp.raise_for_status()

    response_data = resp.json()["responses"][0]

    labels  = response_data.get("labelAnnotations",        [])
    objects = response_data.get("localizedObjectAnnotations", [])
    texts   = response_data.get("textAnnotations",         [])

    return labels, objects, texts


def match_ingredient(text: str) -> str | None:
    """Return the ingredient name if text matches our list, else None."""
    clean = text.lower().strip()
    if clean in VALID_INGREDIENTS and clean not in SKIP_WORDS:
        return clean.capitalize()
    return None


def extract_from_labels(labels: list) -> str | None:
    """Check Vision labels against ingredients list, highest confidence first."""
    for item in labels:
        score = item.get("score", 0)
        if score < 0.7:           # ignore low-confidence labels
            continue
        result = match_ingredient(item["description"])
        if result:
            return result
    return None


def extract_from_objects(objects: list) -> str | None:
    for obj in objects:
        score = obj.get("score", 0)
        if score < 0.6:
            continue
        result = match_ingredient(obj["name"])
        if result:
            return result
    return None


def extract_from_text(texts: list) -> str | None:
  
    if not texts:
        return None

    # texts[0] contains the full raw text from the image
    full_text = texts[0].get("description", "")

    # Split into individual words and check each
    words = re.split(r"[\s\n,./]+", full_text)

    # Also check two-word phrases (e.g. "ice cream", "baked beans")
    phrases = []
    for i in range(len(words) - 1):
        phrases.append(f"{words[i]} {words[i+1]}".lower())

    for phrase in phrases:
        result = match_ingredient(phrase)
        if result:
            return result

    for word in words:
        result = match_ingredient(word)
        if result:
            return result

    return None


def extract_best_food_label(labels, objects, texts) -> str | None:

    # 1. Try objects first — most precise
    result = extract_from_objects(objects)
    if result:
        return result

    # 2. Try OCR text — reads the packaging directly
    result = extract_from_text(texts)
    if result:
        return result

    # 3. Try labels
    result = extract_from_labels(labels)
    if result:
        return result

    # 4. Last resort — first label that isn't a skip word
    for item in labels:
        label = item["description"].lower()
        if label not in SKIP_WORDS:
            return item["description"].capitalize()

    return None


@app.post("/detect")
async def detect_food(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        labels, objects, texts = call_google_vision(image_bytes)

        # Debug log — helpful during development
        print("Labels:",  [(l["description"], round(l.get("score",0), 2)) for l in labels])
        print("Objects:", [(o["name"],        round(o.get("score",0), 2)) for o in objects])
        print("Text:", texts[0]["description"][:100] if texts else "none")

        ingredient = extract_best_food_label(labels, objects, texts)
        return JSONResponse({"ingredient": ingredient})

    except Exception as e:
        print("detect_food error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})