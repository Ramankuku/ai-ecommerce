import re
from pathlib import Path

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / 'products_final.csv'

def _find_index_dir():
    candidates = [
        BASE_DIR / 'data' / 'products.db',   
        BASE_DIR / 'data' / 'faiss_index',
        BASE_DIR / 'data',
    ]
    for folder in candidates:
        if (folder / 'index.faiss').exists() and (folder / 'index.pkl').exists():
            return folder
    return BASE_DIR / 'data' / 'faiss_index'  

DB_PATH = _find_index_dir()

REQUIRED_COLUMNS = [
    "product_id",
    "name",
    "main_category",
    "sub_category",
    "discount_price",
    "actual_price",
    "ratings",
    "no_of_ratings",
    "discount_pct"
]

EMBED_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.5-flash-lite"


IMAGE_DIR = BASE_DIR / 'images'
_IMG_RE = re.compile(r"/images/I/[^?\s]+")

_local_cache = None      
_lookup_cache = None   

def _local_images():
    """File names inside images/. An empty result is never cached, so a
    folder that was not ready at start-up does not stay 'empty' forever."""
    global _local_cache

    if _local_cache:
        return _local_cache

    if not IMAGE_DIR.exists():
        return frozenset()

    _local_cache = frozenset(
        p.name for p in IMAGE_DIR.iterdir() if p.is_file()
    )

    return _local_cache


def _clean(value):
    text = str(value if value is not None else "").strip()
    return "" if text.lower() in ("nan", "none") else text


def _image_lookup():
    global _lookup_cache

    if _lookup_cache is not None:
        return _lookup_cache

    by_id = {}
    by_name = {}

    try:
        df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
        df.columns = [str(c).strip().lower() for c in df.columns]

        for row in df.to_dict("records"):

            entry = (
                _clean(row.get("image_path", "")),
                _clean(row.get("image", ""))
            )

            pid = _clean(row.get("product_id", ""))
            name = _clean(row.get("name", "")).lower()

            if pid:
                by_id[pid] = entry

            if name and name not in by_name:
                by_name[name] = entry

        _lookup_cache = (by_id, by_name)

    except Exception as exc:
        # Do not cache a failure; try again on the next search
        print(f"[utils] Could not build image lookup from CSV: {exc}")
        return {}, {}

    return _lookup_cache


def get_image_url(product_id, image_path="", image="", name=""):

    pid = _clean(product_id)

    by_id, by_name = _image_lookup()

    entry = by_id.get(pid) or by_name.get(_clean(name).lower())

    if entry:
        csv_path, csv_image = entry
        image_path = csv_path or _clean(image_path)
        image = csv_image or _clean(image)
    else:
        image_path = _clean(image_path)
        image = _clean(image)

    local_files = _local_images()
    stored = Path(image_path.replace("\\", "/")).name if image_path else ""

    for file_name in (stored, f"{pid}.jpg"):
        if file_name and file_name in local_files:
            return f"/images/{file_name}"

    match = _IMG_RE.search(image)

    return "https://m.media-amazon.com" + match.group(0) if match else ""
def get_shopping_prompt():

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
You are an intelligent AI-powered e-commerce shopping assistant.

Your job is to help customers discover products and manage their
shopping carts through natural language conversations.

You have access to the following tools:

1. search_products
2. get_product_details
3. prepare_add_to_cart
4. confirm_add_to_cart
5. view_cart
6. remove_from_cart

PRODUCT SEARCH:

- Use search_products to find products.
- Respect brand, category, price and rating requirements.
- Never invent product names, prices, ratings or product IDs.
- Use get_product_details when additional information is required.
- Remember the latest search results.
- Understand references such as first product and second product.

CART MANAGEMENT:

- Use prepare_add_to_cart when the customer requests adding a product.
- Always ask for confirmation before adding a product.
- Only use confirm_add_to_cart after explicit customer approval.
- Cancel the pending operation if the customer declines.
- Use view_cart to retrieve actual cart information.
- Use remove_from_cart to remove products.
- Never claim an operation succeeded without tool confirmation.

CONVERSATION:

- Understand natural language rather than relying on keywords.
- Remember previous conversation context.
- Ask questions when important product information is missing.
- Do not invent information.
- Keep responses concise and conversational.

VOICE:

- Use simple, natural language.
- Avoid technical explanations.
- Respond like a helpful shopping assistant.
"""
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        ("human", "{input}"),

        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        )

    ])

    return prompt