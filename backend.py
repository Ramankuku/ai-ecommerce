import os
import re
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import bcrypt
import jwt
import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "products_final.csv"
IMAGE_DIR = ROOT / "images"
IMAGE_DIR.mkdir(exist_ok=True)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "smartshop"),
}
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-secret-before-deployment")
JWT_ALGORITHM = "HS256"
TOKEN_DAYS = 7


# --------------------------------------------------------------------------- database
def get_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as exc:
        raise HTTPException(status_code=503, detail=f"MySQL connection failed: {exc}")


@contextmanager
def db_cursor(dictionary=True):
    conn = get_db()
    cursor = conn.cursor(dictionary=dictionary)
    try:
        yield conn, cursor
    finally:
        cursor.close()
        conn.close()


def init_db():
    """Create the database and tables. Never stops the API from starting:
    products still load even if MySQL is down (only login/cart need it)."""
    try:
        server_cfg = {k: v for k, v in DB_CONFIG.items() if k != "database"}
        conn = mysql.connector.connect(**server_cfg)
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}`")
        conn.commit()
        cur.close()
        conn.close()
    except mysql.connector.Error as exc:
        print(f"[startup] WARNING - cannot reach MySQL: {exc}")
        return

    statements = [
        """CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(80) PRIMARY KEY,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS cart_items (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            user_id VARCHAR(80) NOT NULL,
            product_id VARCHAR(40) NOT NULL,
            quantity INT NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_user_product (user_id, product_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS orders (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            user_id VARCHAR(80) NOT NULL,
            total_amount DECIMAL(12,2) NOT NULL,
            payment_status VARCHAR(30) NOT NULL DEFAULT 'SUCCESS',
            payment_method VARCHAR(50) NOT NULL DEFAULT 'DEMO',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )""",
        """CREATE TABLE IF NOT EXISTS order_items (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            order_id BIGINT NOT NULL,
            product_id VARCHAR(40) NOT NULL,
            product_name VARCHAR(500) NOT NULL,
            quantity INT NOT NULL,
            actual_price DECIMAL(12,2) NOT NULL,
            discount_price DECIMAL(12,2) NOT NULL,
            line_total DECIMAL(12,2) NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE
        )""",
    ]
    try:
        with db_cursor(dictionary=False) as (conn, cur):
            for statement in statements:
                cur.execute(statement)
            conn.commit()
    except (HTTPException, mysql.connector.Error) as exc:
        print(f"[startup] WARNING - could not create tables: {exc}")


@asynccontextmanager
async def lifespan(_app):
    init_db()
    try:
        products = load_products()
        found = sum(p["image_available"] for p in products)
        print(f"[startup] {len(products)} products loaded, {found} with a local image")
    except HTTPException as exc:
        print(f"[startup] WARNING - {exc.detail}")
    yield


app = FastAPI(title="SmartShop E-commerce API", lifespan=lifespan)
app.mount("/images", StaticFiles(directory=str(IMAGE_DIR), check_dir=False), name="images")


# --------------------------------------------------------------------------- schemas / auth
class SignupRequest(BaseModel):
    user_id: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=8, max_length=100)


class LoginRequest(BaseModel):
    user_id: str
    password: str


class CartRequest(BaseModel):
    product_id: str
    quantity: int = Field(default=1, ge=1, le=20)


class RemoveCartRequest(BaseModel):
    product_id: str


def make_token(user_id: str):
    payload = {"user_id": user_id, "exp": datetime.now(timezone.utc) + timedelta(days=TOKEN_DAYS)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def current_user(authorization: str = Header(default="")) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Please log in.")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return str(payload["user_id"])
    except Exception:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
_IMG_RE = re.compile(r"/images/I/[^?\s]+")


def _number(value):
    try:
        cleaned = str(value).replace(",", "").replace("₹", "").replace("Rs.", "").strip()
        return round(float(cleaned), 2) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0


def _remote_image(url: str) -> str:
    """Repair Amazon 'IMAGERENDERING' links; fall back to the original URL."""
    match = _IMG_RE.search(url or "")
    return "https://m.media-amazon.com" + match.group(0) if match else ""


@lru_cache(maxsize=1)
def load_products():
    if not CSV_PATH.exists():
        raise HTTPException(status_code=500, detail=f"{CSV_PATH.name} was not found beside backend.py.")
    df = pd.read_csv(CSV_PATH)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["product_id", "name", "discount_price", "actual_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"CSV is missing columns: {', '.join(missing)}")

    df = df.fillna("")
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df = df[df["product_id"] != ""].drop_duplicates(subset="product_id")

    local_files = {p.name for p in IMAGE_DIR.iterdir() if p.is_file()}  # one folder scan
    products = []
    for row in df.to_dict("records"):
        pid = row["product_id"]
        stored_name = Path(str(row.get("image_path", "")).replace("\\", "/")).name
        local_name = next((n for n in (stored_name, f"{pid}.jpg") if n and n in local_files), "")

        if local_name:
            image_url = f"/images/{local_name}"
        else:
            image_url = _remote_image(str(row.get("image", "")))

        actual = _number(row.get("actual_price", 0))
        discount = _number(row.get("discount_price", 0))
        pct = _number(row.get("discount_pct", 0))
        if not pct and actual > discount > 0:
            pct = round((actual - discount) / actual * 100, 1)

        products.append({
            "product_id": pid,
            "name": str(row["name"]).strip(),
            "main_category": str(row.get("main_category", "")).strip(),
            "sub_category": str(row.get("sub_category", "")).strip(),
            "discount_price": discount,
            "actual_price": actual,
            "discount_pct": pct,
            "ratings": _number(row.get("ratings", 0)),
            "no_of_ratings": int(_number(row.get("no_of_ratings", 0))),
            "image_url": image_url,                 # relative (/images/..) or a repaired Amazon URL
            "image_available": bool(local_name),
            "product_link": str(row.get("link", "")).strip(),
        })
    return products


@lru_cache(maxsize=1)
def product_index():
    return {p["product_id"]: p for p in load_products()}


@app.get("/")
def root():
    return {"message": "SmartShop API is running"}


@app.get("/health")
def health():
    products = load_products()
    return {
        "status": "ok",
        "products": len(products),
        "local_images": sum(p["image_available"] for p in products),
        "csv": str(CSV_PATH),
        "image_dir": str(IMAGE_DIR),
    }


@app.get("/categories")
def get_categories():
    return {"categories": sorted({p["main_category"] for p in load_products() if p["main_category"]})}


@app.get("/products")
def get_products(search: str = "", category: str = "",
                 page: int = Query(1, ge=1), per_page: int = Query(12, ge=1, le=100)):
    products = load_products()
    if category and category != "All":
        products = [p for p in products if p["main_category"] == category]
    tokens = search.lower().split()
    if tokens:
        def haystack(p):
            return f"{p['name']} {p['main_category']} {p['sub_category']}".lower()
        products = [p for p in products if all(t in haystack(p) for t in tokens)]

    total = len(products)
    pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, pages)
    start = (page - 1) * per_page
    return {"count": total, "page": page, "pages": pages, "products": products[start:start + per_page]}

@app.post("/auth/signup")
def signup(payload: SignupRequest):
    user_id = payload.user_id.strip()
    password_hash = bcrypt.hashpw(payload.password.encode(), bcrypt.gensalt()).decode()
    try:
        with db_cursor() as (conn, cur):
            cur.execute("INSERT INTO users (user_id,password_hash) VALUES (%s,%s)", (user_id, password_hash))
            conn.commit()
        return {"token": make_token(user_id), "user": {"user_id": user_id}}
    except mysql.connector.IntegrityError:
        raise HTTPException(status_code=409, detail="This User ID is already taken.")


@app.post("/auth/login")
def login(payload: LoginRequest):
    user_id = payload.user_id.strip()
    with db_cursor() as (conn, cur):
        cur.execute("SELECT user_id,password_hash FROM users WHERE user_id=%s", (user_id,))
        user = cur.fetchone()
    if not user or not bcrypt.checkpw(payload.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Incorrect User ID or password.")
    return {"token": make_token(user["user_id"]), "user": {"user_id": user["user_id"]}}


# --------------------------------------------------------------------------- cart / orders
@app.get("/cart")
def view_cart(user_id: str = Depends(current_user)):
    products = product_index()
    with db_cursor() as (conn, cur):
        cur.execute("SELECT product_id,quantity FROM cart_items WHERE user_id=%s ORDER BY id DESC", (user_id,))
        rows = cur.fetchall()
    items = []
    for row in rows:
        product = products.get(row["product_id"])
        if product:
            item = dict(product)
            item["quantity"] = row["quantity"]
            item["line_total"] = round(item["discount_price"] * item["quantity"], 2)
            items.append(item)
    return {"items": items, "total": round(sum(i["line_total"] for i in items), 2)}


@app.post("/cart/add")
def add_to_cart(payload: CartRequest, user_id: str = Depends(current_user)):
    if payload.product_id not in product_index():
        raise HTTPException(status_code=404, detail="Product not found.")
    with db_cursor() as (conn, cur):
        cur.execute(
            """INSERT INTO cart_items (user_id,product_id,quantity) VALUES (%s,%s,%s)
               ON DUPLICATE KEY UPDATE quantity=quantity+VALUES(quantity)""",
            (user_id, payload.product_id, payload.quantity),
        )
        conn.commit()
    return {"message": "Product added to cart."}


@app.post("/cart/remove")
def remove_from_cart(payload: RemoveCartRequest, user_id: str = Depends(current_user)):
    with db_cursor() as (conn, cur):
        cur.execute("DELETE FROM cart_items WHERE user_id=%s AND product_id=%s", (user_id, payload.product_id))
        conn.commit()
    return {"message": "Product removed from cart."}


@app.post("/checkout")
def checkout(user_id: str = Depends(current_user)):
    products = product_index()
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    try:
        conn.start_transaction()
        cur.execute("SELECT user_id FROM users WHERE user_id=%s FOR UPDATE", (user_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="User not found.")
        cur.execute("SELECT product_id,quantity FROM cart_items WHERE user_id=%s FOR UPDATE", (user_id,))
        cart = cur.fetchall()
        if not cart:
            raise HTTPException(status_code=400, detail="Your cart is empty.")

        order_lines = []
        total = Decimal("0.00")
        for row in cart:
            product = products.get(row["product_id"])
            if not product:
                continue
            qty = int(row["quantity"])
            actual = Decimal(str(product["actual_price"]))
            discounted = Decimal(str(product["discount_price"]))
            line_total = (discounted * qty).quantize(Decimal("0.01"))
            total += line_total
            order_lines.append((product, qty, actual, discounted, line_total))

        if not order_lines:
            raise HTTPException(status_code=400, detail="No valid products found in your cart.")

        cur.execute(
            """INSERT INTO orders (user_id,total_amount,payment_status,payment_method)
               VALUES (%s,%s,'SUCCESS','DEMO')""",
            (user_id, total),
        )
        order_id = cur.lastrowid
        for product, qty, actual, discounted, line_total in order_lines:
            cur.execute(
                """INSERT INTO order_items
                   (order_id,product_id,product_name,quantity,actual_price,discount_price,line_total)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (order_id, product["product_id"], product["name"], qty, actual, discounted, line_total),
            )
        cur.execute("DELETE FROM cart_items WHERE user_id=%s", (user_id,))
        conn.commit()
        return {"message": "Payment successful (demo payment).", "order_id": order_id,
                "total_amount": float(total), "payment_status": "SUCCESS"}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Checkout failed: {exc}")
    finally:
        cur.close()
        conn.close()


@app.get("/orders")
def order_history(user_id: str = Depends(current_user)):
    with db_cursor() as (conn, cur):
        cur.execute(
            """SELECT id,user_id,total_amount,payment_status,created_at
               FROM orders WHERE user_id=%s ORDER BY id DESC""", (user_id,)
        )
        orders = cur.fetchall()
        for order in orders:
            cur.execute(
                """SELECT product_id,product_name,quantity,actual_price,discount_price,line_total
                   FROM order_items WHERE order_id=%s ORDER BY id""", (order["id"],)
            )
            order["items"] = cur.fetchall()
    return {"orders": orders}