import os
import re
import json
import requests

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from src.utils import get_image_url, LLM_MODEL

load_dotenv()


SYSTEM_PROMPT = """You are SmartShop, a friendly shopping assistant.

- To find products, call search_products. Keep your reply to 1-2 short sentences.
  Product cards are shown to the customer automatically, so do not list them.
- When the customer says "add product N to cart" (or similar), call add_to_cart
  IMMEDIATELY. Never ask for confirmation.
- Use view_cart / remove_from_cart for cart questions.
- Use get_product_details to compare or answer questions about a searched product.
- Use recommend_best_product when the customer asks which product is best.
- Never invent products, prices or ratings.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

CART_ACTIONS = {"view_cart", "remove_from_cart", "add_to_cart"}


class VoiceSearch:

    def __init__(self, rag, state):

        self.rag = rag
        self.state = state

        self.api_url = os.getenv(
            "API_URL",
            "http://127.0.0.1:8000"
        ).rstrip("/")

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.2,
            api_key=os.getenv('GOOGLE_API_KEY')
        )

        for key, default in (
            ("search_results", []),
            ("chat_history", []),
            ("display_history", []),
            ("speech", ""),
        ):
            self.state.setdefault(key, default)

        self.tools = self.create_tools()

        agent = create_tool_calling_agent(
            self.llm,
            self.tools,
            PROMPT
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            max_iterations=6,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def api_request(self, method, endpoint, **kwargs):

        token = self.state.get("token")

        if not token:
            raise ValueError(
                "Please log in before managing your cart."
            )

        response = requests.request(
            method,
            f"{self.api_url}{endpoint}",
            headers={
                "Authorization": f"Bearer {token}"
            },
            timeout=20,
            **kwargs,
        )

        if response.status_code >= 400:

            try:
                detail = response.json().get(
                    "detail",
                    response.text
                )
            except Exception:
                detail = response.text

            raise RuntimeError(str(detail))

        return (
            response.json()
            if response.content
            else {"message": "Done."}
        )

    def backend_image_url(self, product_id, name):

        try:
            response = requests.get(
                f"{self.api_url}/products",
                params={
                    "search": str(name or ""),
                    "per_page": 100
                },
                timeout=10,
            )

            response.raise_for_status()

            for product in response.json().get("products", []):

                if str(product.get("product_id")) == str(product_id):
                    return product.get("image_url", "")

        except Exception as error:
            print(
                f"[voice] Image lookup failed for {product_id}: {error}"
            )

        return ""

    def _product(self, number):

        products = self.state.get("search_results", [])

        if 1 <= number <= len(products):
            return products[number - 1]

        return None
    def create_tools(self):

        @tool
        def search_products(query: str, k: int = 5):
            """Search products by name, category, price or rating."""

            products = []

            results = self.rag.search_products(
                query,
                k=k
            )

            for i, result in enumerate(results, start=1):

                metadata = result["document"].metadata

                product = {
                    "product_number": i,
                    "product_id": metadata.get("product_id"),
                    "product_name": metadata.get("name"),
                    "brand": metadata.get("brand"),
                    "category": metadata.get("main_category"),
                    "sub_category": metadata.get("sub_category"),
                    "discount_price": metadata.get("discount_price"),
                    "actual_price": metadata.get("actual_price"),
                    "ratings": metadata.get("ratings"),
                    "no_of_ratings": metadata.get("no_of_ratings"),
                    "discount_pct": metadata.get("discount_pct"),

                    "image_url": (
                        self.backend_image_url(
                            metadata.get("product_id"),
                            metadata.get("name")
                        )
                        or get_image_url(
                            metadata.get("product_id"),
                            metadata.get("image_path", ""),
                            metadata.get("image", "")
                        )
                    ),
                }

                products.append(product)

            self.state["search_results"] = products

            if not products:
                return "No matching products were found."

            slim = [
                {
                    key: value
                    for key, value in product.items()
                    if key != "image_url"
                }
                for product in products
            ]

            return json.dumps(
                slim,
                ensure_ascii=False,
                default=str
            )

        @tool
        def get_product_details(product_number: int):
            """Get details of a product from the latest search results."""

            product = self._product(product_number)

            if not product:
                return "Product number is not available. Search again."

            return json.dumps(
                product,
                ensure_ascii=False,
                default=str
            )

        @tool
        def add_to_cart(product_number: int, quantity: int = 1):
            """Add a product from the latest search results to the cart."""

            product = self._product(product_number)

            if not product:
                return "Invalid product number. Search for the product again."

            if not 1 <= quantity <= 20:
                return "Quantity must be between 1 and 20."

            try:
                self.api_request(
                    "POST",
                    "/cart/add",
                    json={
                        "product_id": str(product["product_id"]),
                        "quantity": int(quantity),
                    }
                )

            except Exception as error:
                return f"Could not add product to cart: {error}"

            return (
                f"Added {quantity} x "
                f"{product['product_name']} to the cart."
            )

        @tool
        def view_cart():
            try:
                return json.dumps(
                    self.api_request("GET", "/cart"),
                    ensure_ascii=False,
                    default=str
                )

            except Exception as error:
                return f"Unable to retrieve cart: {error}"

        @tool
        def remove_from_cart(product_id: str):
            try:
                return json.dumps(
                    self.api_request(
                        "POST",
                        "/cart/remove",
                        json={
                            "product_id": str(product_id)
                        }
                    ),
                    ensure_ascii=False,
                    default=str
                )

            except Exception as error:
                return f"Unable to remove product: {error}"

        @tool
        def recommend_best_product(user_need: str):
            """
            Recommend the best product from the latest search results.

            Use when the customer asks:
            - Which is best?
            - Which should I buy?
            - Recommend one.
            - Best value for money.
            - Best option for my needs.
            """

            products = self.state.get("search_results", [])

            if not products:
                return "No products available. Please search first."

            product_text = []

            for product in products:

                product_text.append(
                    f"""
                    Product Number: {product.get('product_number')}
                    Name: {product.get('product_name')}
                    Brand: {product.get('brand')}
                    Price: {product.get('discount_price')}
                    Rating: {product.get('ratings')}
                    Number of Ratings: {product.get('no_of_ratings')}
                    Category: {product.get('category')}
                    """
                )

            prompt = f"""
            User Requirement:
            {user_need}

            Available Products:
            {chr(10).join(product_text)}

            Analyze the available products and recommend the best option.

            Consider:
            - Product rating
            - Number of ratings
            - Value for money
            - Suitability for the user's requirement

            Explain:
            1. Which product is best.
            2. Why it is best.
            3. Alternative options if available.

            Only use the supplied product information.
            Do not invent specifications.
            Keep your response under 150 words.
            """

            response = self.llm.invoke(prompt)

            return self.to_text(response.content)

        return [
            search_products,
            get_product_details,
            add_to_cart,
            view_cart,
            remove_from_cart,
            recommend_best_product,
        ]

    @staticmethod
    def to_text(output):
        """Convert Gemini output into a normal string."""

        if isinstance(output, str):
            return output

        if isinstance(output, list):

            return "".join(
                str(item.get("text", ""))
                if isinstance(item, dict)
                else str(item)
                for item in output
            ).strip()

        if isinstance(output, dict):
            return str(output.get("text", output))

        return str(output)

    def chat(self, query):

        history = self.state["chat_history"]

        response = self.agent_executor.invoke({
            "input": query,
            "chat_history": history[-10:],
        })

        answer = self.to_text(response["output"])

        self.state["chat_history"] = (
            history
            + [
                HumanMessage(content=query),
                AIMessage(content=answer)
            ]
        )[-20:]

        used = {
            step[0].tool
            for step in response.get("intermediate_steps", [])
        }

        products = (
            []
            if used & CART_ACTIONS
            else list(self.state["search_results"])
        )
        added = any(
            step[0].tool == "add_to_cart"
            and str(step[1]).startswith("Added")
            for step in response.get("intermediate_steps", [])
        )
        if added:
            speech = "Added to your cart."

        elif "search_products" in used:

            count = len(self.state["search_results"])

            speech = (
                f"I found {count} product{'s' if count != 1 else ''} for you."
                if count
                else "Sorry, I could not find any matching products."
            )

        else:
            speech = re.sub(r"[*_#`>|]", "", answer).strip()[:400]

        self.state["speech"] = speech

        self.state["display_history"].extend([
            {
                "role": "user",
                "content": query,
                "products": []
            },
            {
                "role": "assistant",
                "content": answer,
                "products": products
            }
        ])

        return answer