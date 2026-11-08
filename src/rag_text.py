import os
import json
import time
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import pandas as pd
from dotenv import load_dotenv
from src.utils import CSV_PATH, DB_PATH, REQUIRED_COLUMNS, LLM_MODEL

load_dotenv()

class TextRAG:
    def __init__(self):
        self.csv_path = Path(CSV_PATH)
        self.db_path = Path(DB_PATH)
        self.vector_store = None
        self.brands = set()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.2,
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
        self.filter_prompt = ChatPromptTemplate.from_template("""
You are an e-commerce search query parser.

Extract the filters from the customer's query.

Return ONLY valid JSON. Do not include markdown or explanations.

Customer Query:
{query}

Use exactly this JSON structure:

{{
    "product_type": null,
    "category": null,
    "brand": null,
    "min_price": null,
    "max_price": null,
    "min_rating": null,
    "sort_by": null
}}

Rules:
- product_type: Product requested, such as jeans, phone, shoes or laptop.
- category: Main product category, if identifiable.
- brand: Brand name, if mentioned.
- min_price: Minimum price in INR, if specified.
- max_price: Maximum price in INR, if specified.
- min_rating: Minimum customer rating, if specified.
- sort_by: Use "price" for cheapest products or "rating" for highest-rated products.
- Use null for unspecified filters.

Example:

Query: Show me jeans under 5000

{{
    "product_type": "jeans",
    "category": null,
    "brand": null,
    "min_price": null,
    "max_price": 5000,
    "min_rating": null,
    "sort_by": null
}}

Query: Samsung phones with rating above 4

{{
    "product_type": "phone",
    "category": "electronics",
    "brand": "Samsung",
    "min_price": null,
    "max_price": null,
    "min_rating": 4,
    "sort_by": null
}}
""")

        self.filter_chain = (
            self.filter_prompt
            | self.llm
            | StrOutputParser()
        )
        self.answer_prompt = ChatPromptTemplate.from_template("""
You are an intelligent e-commerce shopping assistant.

Answer the customer's question using ONLY the retrieved product information.

Customer Question:
{question}

Retrieved Products:
{context}

Instructions:

1. Recommend only products present in the retrieved information.
2. Never invent product names, prices, ratings or specifications.
3. Include prices and ratings when available.
4. Explain briefly why the products match the customer's requirements.
5. Compare products if the customer asks for a comparison.
6. If no products are provided, say that no matching products were found.
7. Give a clear and helpful recommendation.
8. Do not mention FAISS, embeddings or internal technical details.

Write a concise and useful shopping recommendation.
""")

        self.answer_chain = (
            self.answer_prompt
            | self.llm
            | StrOutputParser()
        )
    def extract_brand(self, product_name):

        words = str(product_name).strip().split()

        if not words:
            return ""

        return " ".join(words[:2]).strip(".,-()[]").lower()
    def load_products(self):

        documents = []

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Product CSV not found: {self.csv_path}"
            )

        df = pd.read_csv(self.csv_path)

        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing CSV columns: {missing_columns}"
            )

        for _, row in df.iterrows():

            discount_price = pd.to_numeric(
                row["discount_price"], errors="coerce"
            )

            actual_price = pd.to_numeric(
                row["actual_price"], errors="coerce"
            )

            rating = pd.to_numeric(
                row["ratings"], errors="coerce"
            )

            rating_count = pd.to_numeric(
                row["no_of_ratings"], errors="coerce"
            )

            discount_percentage = pd.to_numeric(
                row["discount_pct"], errors="coerce"
            )

            name = str(row["name"])
            main_category = str(row["main_category"])
            sub_category = str(row["sub_category"])

            brand = self.extract_brand(name)

            if brand:
                self.brands.add(brand)

            page_content = f"""
Product Name: {name}
Main Category: {main_category}
Subcategory: {sub_category}
Brand: {brand if brand else 'Not available'}

Selling Price: {
    discount_price if pd.notna(discount_price) else 'Not available'
} INR

Original Price: {
    actual_price if pd.notna(actual_price) else 'Not available'
} INR

Customer Rating: {
    rating if pd.notna(rating) else 'Not available'
} out of 5

Number of Ratings: {
    rating_count if pd.notna(rating_count) else 'Not available'
}

Discount: {
    discount_percentage if pd.notna(discount_percentage) else 'Not available'
} percent
"""

            metadata = {
                "product_id": str(row["product_id"]),
                "name": name,
                "main_category": main_category,
                "sub_category": sub_category,

                "source_file": str(row.get("source_file", "")),
                "image": str(row.get("image", "")),
                "link": str(row.get("link", "")),
                "image_path": str(row.get("image_path", "")),
                "image_status": str(row.get("image_status", "")),

                "brand": brand,

                "discount_price": (
                    float(discount_price)
                    if pd.notna(discount_price) else -1.0
                ),

                "actual_price": (
                    float(actual_price)
                    if pd.notna(actual_price) else -1.0
                ),

                "ratings": (
                    float(rating)
                    if pd.notna(rating) else -1.0
                ),

                "no_of_ratings": (
                    int(rating_count)
                    if pd.notna(rating_count) else 0
                ),

                "discount_pct": (
                    float(discount_percentage)
                    if pd.notna(discount_percentage) else 0.0
                )
            }

            documents.append(
                Document(
                    page_content=page_content.strip(),
                    metadata=metadata
                )
            )

        return documents

    def extract_filters(self, query):

        default_filters = {
            "product_type": None,
            "category": None,
            "brand": None,
            "min_price": None,
            "max_price": None,
            "min_rating": None,
            "sort_by": None
        }

        try:

            response = self.filter_chain.invoke({
                "query": query
            })

            response = response.strip()
            response = response.replace("```json", "").replace("```", "").strip()

            filters = json.loads(response)

            for key in default_filters:
                if key in filters:
                    default_filters[key] = filters[key]

            return default_filters

        except Exception as exc:
            
            print(f"[rag] Filter extraction failed, using no filters: {exc}")

            return default_filters

    def filter_products(self, results, filters):

        filtered_products = []

        for doc, distance in results:

            metadata = doc.metadata

            name = str(metadata["name"]).lower()
            category = str(metadata["main_category"]).lower()
            sub_category = str(metadata["sub_category"]).lower()
            brand = str(metadata["brand"]).lower()

            price = metadata["discount_price"]
            rating = metadata["ratings"]

            if filters["product_type"]:

                product_type = str(
                    filters["product_type"]
                ).lower()

                if (
                    product_type not in name
                    and product_type not in sub_category
                ):
                    continue

            if filters["category"]:

                requested_category = str(
                    filters["category"]
                ).lower()

                if (
                    requested_category not in category
                    and requested_category not in sub_category
                ):
                    continue

            if filters["brand"]:

                requested_brand = str(
                    filters["brand"]
                ).lower()

                if (
                    requested_brand not in brand
                    and requested_brand not in name
                ):
                    continue

            if filters["min_price"] is not None:

                if (
                    price < 0
                    or price < float(filters["min_price"])
                ):
                    continue

            if filters["max_price"] is not None:

                if (
                    price < 0
                    or price > float(filters["max_price"])
                ):
                    continue

            if filters["min_rating"] is not None:

                if (
                    rating < 0
                    or rating < float(filters["min_rating"])
                ):
                    continue

            filtered_products.append({
                "document": doc,
                "distance": float(distance),
                "product_id": metadata.get("product_id"),
                "name": metadata.get("name"),
                "image": metadata.get("image"),
                "image_path": metadata.get("image_path"),
                "link": metadata.get("link")
            })

        return filtered_products

    def create_vectorstore(self):
        
        if self.vector_store is not None:
            return self.vector_store

        index_file = self.db_path / "index.faiss"
        pickle_file = self.db_path / "index.pkl"

        if index_file.exists() and pickle_file.exists():

            print("\nLoading existing FAISS vector store...")

            self.vector_store = FAISS.load_local(
                str(self.db_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            print("FAISS vector store loaded successfully.")
            return self.vector_store

        # Nothing saved yet -> build once and save
        print("\nCreating a new FAISS vector store...")

        documents = self.load_products()

        if not documents:
            raise ValueError("No products found in CSV.")

        print(f"Total products: {len(documents)}")

        self.vector_store = FAISS.from_documents(
            documents,
            embedding=self.embeddings
        )
        self.db_path.mkdir(
            parents=True,
            exist_ok=True
        )

        self.vector_store.save_local(
            str(self.db_path)
        )

        return self.vector_store

    def search_products(self, query, k=5):

        query = query.strip()

        if not query:
            return []

        filters = self.extract_filters(query)

        vector_store = self.create_vectorstore()
        results = vector_store.similarity_search_with_score(
            query,
            k=vector_store.index.ntotal
        )

        filtered_results = self.filter_products(
            results,
            filters
        )
        filtered_results.sort(
            key=lambda item: item["distance"]
        )
        if filters["sort_by"] == "price":

            filtered_results.sort(
                key=lambda item: item["document"].metadata["discount_price"]
            )

        elif filters["sort_by"] == "rating":

            filtered_results.sort(
                key=lambda item: item["document"].metadata["ratings"],
                reverse=True
            )

        return filtered_results[:k]

    def ask_question(self, question):

        question = question.strip()

        search_results = self.search_products(
            query=question,
            k=5
        )

        if not search_results:

            return {
                "answer": "No matching products found.",
                "products": []
            }

        context = ""
        products = []

        for index, result in enumerate(
            search_results,
            start=1
        ):

            doc = result["document"]
            metadata = doc.metadata

            product = {
                "product_id": metadata.get("product_id"),
                "name": metadata.get("name"),
                "brand": metadata.get("brand"),
                "category": metadata.get("main_category"),
                "sub_category": metadata.get("sub_category"),
                "price": metadata.get("discount_price"),
                "actual_price": metadata.get("actual_price"),
                "rating": metadata.get("ratings"),
                "no_of_ratings": metadata.get("no_of_ratings"),
                "discount_pct": metadata.get("discount_pct"),
                "image": metadata.get("image"),
                "image_path": metadata.get("image_path"),
                "link": metadata.get("link")
            }

            products.append(product)

            context += f"""
    Product {index}

    Name: {metadata.get('name')}
    Brand: {metadata.get('brand')}
    Category: {metadata.get('main_category')}
    Subcategory: {metadata.get('sub_category')}

    Selling Price: ₹{metadata.get('discount_price')}
    Original Price: ₹{metadata.get('actual_price')}

    Rating: {metadata.get('ratings')} out of 5

    Number of Ratings:
    {metadata.get('no_of_ratings')}

    Discount:
    {metadata.get('discount_pct')}%

    Product Link:
    {metadata.get('link')}

    Product Information:
    {doc.page_content}

    """

        try:

            answer = self.answer_chain.invoke({
                "question": question,
                "context": context
            })

            return {
                "answer": answer,
                "products": products
            }

        except Exception as e:

            return {
                "answer": f"Error generating response: {e}",
                "products": products
            }