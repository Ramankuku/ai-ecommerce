import os
import html
import json
import time
import requests
import streamlit as st
import streamlit.components.v1 as components

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mic_recorder import speech_to_text

from src import history
from src.rag_text import TextRAG
from src.voice_search import VoiceSearch

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PER_PAGE = 12

st.set_page_config(page_title="SmartShop", page_icon="🛍️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {background: linear-gradient(135deg,#1f3c88,#4b7bec); color: #fff;
       padding: 22px 28px; border-radius: 14px; margin-bottom: 16px;}
.hero h1 {margin: 0; font-size: 30px; color: #fff; padding: 0;}
.hero p {margin: 4px 0 0; opacity: .9; font-size: 15px;}
.price {font-size: 20px; font-weight: 700;}
.old-price {color: #888; text-decoration: line-through; font-size: 13px;}
.pname {font-weight: 600; font-size: 14px; line-height: 1.3; height: 2.6em; overflow: hidden;}
.badge {background: #e8f5e9; color: #1b7f3b; font-size: 12px; font-weight: 600;
        padding: 2px 8px; border-radius: 10px; margin-left: 6px;}
[data-testid="stImage"] img {object-fit: contain; max-height: 190px;}
</style>
""", unsafe_allow_html=True)

def api(method, path, payload=None, auth=True, params=None):
    headers = {}
    if auth and st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        r = requests.request(method, f"{API_URL}{path}", json=payload,
                             headers=headers, params=params, timeout=30)
    except requests.RequestException as exc:
        st.error(f"Cannot connect to backend at {API_URL}. Start FastAPI first. ({exc})")
        return None

    if r.status_code >= 400:
        try:
            st.error(str(r.json().get("detail", r.text)))
        except Exception:
            st.error(r.text)
        return None
    return r.json() if r.content else {}


def img_src(url):
    return f"{API_URL}{url}" if url and url.startswith("/") else (url or "")


def money(value):
    return f"₹{float(value):,.0f}"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_categories(api_url):
    r = requests.get(f"{api_url}/categories", timeout=30)
    r.raise_for_status()
    return r.json()["categories"]


@st.cache_resource(show_spinner=False)
def get_rag():
    rag = TextRAG()
    rag.create_vectorstore()
    return rag


@st.cache_resource(show_spinner=False)
def get_image_search():
    from src.image_search import ImageSearch
    return ImageSearch()


def speak(text):
    """Speak a short sentence in the browser (Web Speech API)."""
    if text:
        components.html(
            f"<script>/*{time.time()}*/(function(){{"
            f"var w=window;try{{if(window.parent.speechSynthesis)w=window.parent}}catch(e){{}}"
            f"var u=new w.SpeechSynthesisUtterance({json.dumps(text)});"
            f"w.speechSynthesis.cancel();w.speechSynthesis.speak(u);}})();</script>",
            height=0)


def product_card(p, key):
    name = p.get("product_name") or p.get("name", "")
    category = " • ".join(str(x) for x in (
        p.get("category") or p.get("main_category"), p.get("sub_category")) if x)
    local = p.get("local_path")
    image = local if local and os.path.exists(local) else img_src(p.get("image_url"))
    price, actual = p.get("discount_price"), p.get("actual_price")
    rating = p.get("ratings")

    with st.container(border=True):
        if image:
            st.image(image)
        else:
            st.caption("No image available")

        number = f"{p['product_number']}. " if p.get("product_number") else ""
        st.markdown(f"<div class='pname'>{number}{html.escape(name[:80])}</div>",
                    unsafe_allow_html=True)
        st.caption(category)
        if p.get("similarity") is not None:
            st.caption(f"Match: {p['similarity']}%")

        if price and price > 0:
            old = (f"<span class='old-price'>{money(actual)}</span>"
                   if actual and actual > price else "")
            badge = (f"<span class='badge'>{p['discount_pct']:.0f}% off</span>"
                     if p.get("discount_pct") else "")
            st.markdown(f"<span class='price'>{money(price)}</span> {old}{badge}",
                        unsafe_allow_html=True)

        if rating and rating > 0:
            count = f" ({p['no_of_ratings']:,} ratings)" if p.get("no_of_ratings") else ""
            st.caption(f"⭐ {rating:.1f}{count}")

        if st.button("Add to cart", key=f"{key}_{p['product_id']}", use_container_width=True):
            if api("POST", "/cart/add", {"product_id": str(p["product_id"]), "quantity": 1}):
                st.toast("Added to cart ✅")


def product_grid(products, key, per_row):
    for start in range(0, len(products), per_row):
        row = products[start:start + per_row]
        for col, p in zip(st.columns(per_row), row):
            with col:
                product_card(p, key)

def render_history():
    """One button per saved conversation; click to reopen it."""
    for chat in st.session_state.history[:10]:
        st.button(chat["title"], key=f"hist_{chat['id']}", help=chat["at"],
                  use_container_width=True, on_click=open_chat, args=(chat["id"],),
                  type="primary" if chat["id"] == st.session_state.chat_id else "secondary")

def new_chat():
    s = st.session_state
    s.display_history, s.chat_history, s.search_results = [], [], []
    s.chat_id, s.speech = None, ""


def open_chat(chat_id):
    s = st.session_state
    chat = next(c for c in s.history if c["id"] == chat_id)
    msgs = chat["messages"]
    s.display_history = msgs
    s.chat_history = [
        (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
        for m in msgs][-20:]
    s.search_results = next(
        (m["products"] for m in reversed(msgs) if m["role"] == "assistant" and m["products"]), [])
    s.chat_id, s.speech = chat_id, ""


def save_current():
    s = st.session_state
    if s.display_history:
        s.chat_id = s.chat_id or str(int(time.time() * 1000))
        s.history = history.save(s.user["user_id"], s.chat_id, s.display_history)


def clear_history():
    st.session_state.history = history.clear(st.session_state.user["user_id"])


def process_pending():
    """Run any waiting text / voice / image request."""
    query, image = st.session_state.pending_query, st.session_state.pending_image
    st.session_state.pending_query, st.session_state.pending_image = "", None
    if not query and not image:
        return

    try:
        if image:
            with st.spinner("Finding similar products..."):
                run_image_search(image)
        if query:
            with st.spinner("AI is thinking..."):
                st.session_state.voice_agent.chat(query)
            st.session_state.agent_error = ""
    except Exception as error:
        st.session_state.agent_error = f"Agent error: {error}"
    save_current()


def run_image_search(image_bytes):
    matches = get_image_search().search(image_bytes, k=5)
    products = [{
        "product_number": i, "product_id": m["product_id"],
        "product_name": m["name"], "category": m["category"],
        "sub_category": m["sub_category"], "discount_price": m["price"],
        "ratings": m["rating"], "image_url": m.get("image", ""),
        "local_path": m.get("local_image_path", ""), "similarity": m["similarity"],
    } for i, m in enumerate(matches, start=1)]

    answer = (f"I found {len(products)} products similar to your image. "
              "Tell me a product number to add it to your cart."
              if products else "I could not find similar products for this image.")

    s = st.session_state
    s.search_results = products
    s.display_history += [
        {"role": "user", "content": "📷 Image search", "image": image_bytes, "products": []},
        {"role": "assistant", "content": answer, "products": products},
    ]
    s.chat_history = (s.chat_history + [
        HumanMessage(content="I searched by uploading a product image."),
        AIMessage(content=answer)])[-20:]
    s.speech = (f"I found {len(products)} similar products." if products
                else "Sorry, I could not find similar products.")
    
DEFAULTS = {
    "token": "", "user": None, "page_no": 1, "filters": None, "rag": None,
    "voice_agent": None, "chat_history": [], "display_history": [],
    "search_results": [], "pending_query": "", "pending_image": None,
    "voice_counter": 0, "image_counter": 0, "agent_error": "",
    "speech": "", "history": None, "chat_id": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.markdown("""
<div class="hero"><h1>🛍️ SmartShop</h1>
<p>Search, speak or snap a photo — your AI shopping assistant finds it and adds it to your cart.</p></div>
""", unsafe_allow_html=True)

if not st.session_state.token:
    for tab, (title, path, label) in zip(
        st.tabs(["Login", "Sign up"]),
        [("login", "/auth/login", "Login"), ("signup", "/auth/signup", "Create account")],
    ):
        with tab, st.form(f"{title}_form"):
            uid = st.text_input("User ID", key=f"{title}_uid")
            pwd = st.text_input("Password", type="password", key=f"{title}_pwd")
            confirm = (st.text_input("Confirm password", type="password")
                       if title == "signup" else pwd)
            if st.form_submit_button(label):
                if pwd != confirm:
                    st.error("Passwords do not match.")
                elif result := api("POST", path, {"user_id": uid, "password": pwd}, auth=False):
                    st.session_state.token = result["token"]
                    st.session_state.user = result["user"]
                    st.rerun()
    st.stop()

user_id = st.session_state.user["user_id"]

if st.session_state.history is None:        
    st.session_state.history = history.load(user_id)

try:
    if st.session_state.rag is None:
        with st.spinner("Loading product search system (first run may take a few minutes)..."):
            st.session_state.rag = get_rag()
    if st.session_state.voice_agent is None:
        st.session_state.voice_agent = VoiceSearch(
            rag=st.session_state.rag, state=st.session_state)
except Exception as error:
    st.error(f"AI assistant initialization failed: {error}")


with st.sidebar:
    st.write(f"**Hello, {user_id}**")
    page = st.radio("Navigate", ["Shop", "Cart", "My orders"], key="navigation")

    st.divider()
    st.button("➕ New chat", on_click=new_chat, use_container_width=True)
    st.markdown("##### 🕘 Recent chats")
    if st.session_state.history:
        render_history()
        st.button("Clear history", on_click=clear_history, use_container_width=True)
    else:
        st.caption("Your chats will appear here.")

    st.divider()
    if st.button("Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    with st.expander("Backend status"):
        if health := api("GET", "/health", auth=False):
            st.write(f"Products loaded: **{health['products']}**")
            st.write(f"With local image: **{health['local_images']}**")

if page == "Shop":
    c_search, c_voice, c_image, c_cat = st.columns([4.5, 1.8, 1.3, 2])

    search = c_search.text_input(
        "Search", placeholder="🔍 Search products by name or category...",
        label_visibility="collapsed", key="catalogue_search")

    with c_voice:
        voice_text = speech_to_text(
            language="en", start_prompt="🎤 Voice", stop_prompt="⏹ Stop",
            just_once=True, use_container_width=True,
            key=f"voice_{st.session_state.voice_counter}")
        if voice_text:
            st.session_state.pending_query = voice_text
            st.session_state.voice_counter += 1
            st.toast(f"🎤 Heard: {voice_text}")
            st.rerun()

    with c_image, st.popover("📷 Image", use_container_width=True):
        st.caption("Upload a photo to find similar products.")
        uploaded = st.file_uploader(
            "Upload a product photo", type=["jpg", "jpeg", "png", "webp"],
            key=f"image_{st.session_state.image_counter}", label_visibility="collapsed")
        if uploaded:
            st.session_state.pending_image = uploaded.getvalue()
            st.session_state.image_counter += 1
            st.rerun()

    try:
        categories = ["All"] + fetch_categories(API_URL)
    except requests.RequestException as exc:
        categories = ["All"]
        st.error(f"Could not load categories: {exc}")
    category = c_cat.selectbox("Category", categories, label_visibility="collapsed")

    process_pending()
    if st.session_state.speech:
        speak(st.session_state.speech)
        st.session_state.speech = ""
    if st.session_state.agent_error:
        st.error(st.session_state.agent_error)
    chat = st.session_state.display_history
    if chat:
        with st.container(border=True):
            h1, h2 = st.columns([6, 1])
            h1.markdown("#### 🤖 AI Shopping Assistant")
            h2.button("➕ New chat", on_click=new_chat, use_container_width=True,
                      key="panel_new_chat")

            for p in range(len(chat) - 2, max(0, len(chat) - 6) - 1, -2):
                for i in (p, p + 1):
                    msg = chat[i]
                    with st.chat_message(msg["role"]):
                        if msg.get("image"):
                            st.image(msg["image"], width=150)
                        st.markdown(msg["content"])
                        if msg["role"] == "assistant":
                            product_grid(msg["products"], f"m{i}", 5)
    else:
        st.caption("💡 Ask the AI below, tap 🎤 Voice or 📷 Image — e.g. “show me jeans under 1000”, "
                   "“add product 2 to cart”.")

    st.markdown("### Browse catalogue")
    if st.session_state.filters != (search, category):
        st.session_state.filters = (search, category)
        st.session_state.page_no = 1

    result = api("GET", "/products", auth=False, params={
        "search": search, "category": category,
        "page": st.session_state.page_no, "per_page": PER_PAGE})

    if result is None:
        st.warning("Product list could not be loaded.")
    elif not result["products"]:
        st.info("No products found.")
    else:
        st.session_state.page_no = page_no = result["page"]
        st.caption(f"{result['count']} products found • page {page_no} of {result['pages']}")
        product_grid(result["products"], "cat", 4)

        prev_col, _, next_col = st.columns([1, 2, 1])
        if prev_col.button("◀ Previous", disabled=page_no <= 1):
            st.session_state.page_no -= 1
            st.rerun()
        if next_col.button("Next ▶", disabled=page_no >= result["pages"]):
            st.session_state.page_no += 1
            st.rerun()

    if prompt := st.chat_input("Ask the AI assistant — search, compare, add to cart..."):
        st.session_state.pending_query = prompt
        st.rerun()

elif page == "Cart":
    st.subheader("Your shopping cart")
    cart = api("GET", "/cart")
    items = cart["items"] if cart else []

    if not items:
        st.info("Your cart is empty. Add products from the Shop page.")
    else:
        for item in items:
            c1, c2, c3 = st.columns([1, 4, 2])
            if item.get("image_url"):
                c1.image(img_src(item["image_url"]), width=100)
            c2.markdown(f"**{item['name']}**")
            c2.caption(f"Product ID: {item['product_id']} | Quantity: {item['quantity']}")
            c2.write(f"Actual: ₹{item['actual_price']:,.2f} | Discounted: ₹{item['discount_price']:,.2f}")
            c3.markdown(f"**₹{item['line_total']:,.2f}**")
            if c3.button("Remove", key=f"rm_{item['product_id']}"):
                if api("POST", "/cart/remove", {"product_id": item["product_id"]}):
                    st.rerun()

        st.divider()
        st.subheader(f"Total payable: ₹{cart['total']:,.2f}")
        st.warning("Demo checkout: no real money is charged.")
        if st.button("Pay Now", type="primary") and (order := api("POST", "/checkout")):
            st.success(f"Payment successful! Your order number is #{order['order_id']}.")
            st.balloons()

elif page == "My orders":
    st.subheader("Your purchase history")
    data = api("GET", "/orders")
    orders = data["orders"] if data else []
    if not orders:
        st.info("You have not placed an order yet.")

    for o in orders:
        with st.expander(f"Order #{o['id']} • ₹{float(o['total_amount']):,.2f} • {o['created_at']}"):
            st.write(f"Payment status: {o['payment_status']}")
            for it in o["items"]:
                st.write(f"- {it['product_name']} | Qty: {it['quantity']} | "
                         f"Line total: ₹{float(it['line_total']):,.2f}")