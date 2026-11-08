import io
import json
import hashlib
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from src.utils import CSV_PATH, IMAGE_DIR

class ImageSearch:
    def __init__(self):
        self.csv_path = Path(CSV_PATH)
        self.image_dir = Path(IMAGE_DIR)
        self.cache_dir = self.csv_path.parent / "data" / "image_search"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "image_index.faiss"
        self.ids_path = self.cache_dir / "product_ids.json"
        self.metadata_path = self.cache_dir / "metadata.json"
        self.model_name = "clip-ViT-B-32"
        self.model = SentenceTransformer(self.model_name)
        self.products = {}
        self.product_ids = []
        self.index = None

        self.load_products()
        self.load_or_create_index()

    def load_products(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path).fillna("")

        for _, row in df.iterrows():

            product_id = str(row["product_id"])

            image_path = self.get_image_path(
                row.get("image_path", ""),
                product_id
            )

            try:
                price = float(row["discount_price"])
            except (ValueError, TypeError):
                price = 0.0

            try:
                rating = float(row["ratings"])
            except (ValueError, TypeError):
                rating = 0.0

            self.products[product_id] = {
                "product_id": product_id,
                "name": str(row["name"]),
                "category": str(row["main_category"]),
                "sub_category": str(row["sub_category"]),
                "price": price,
                "rating": rating,
                "image": str(row.get("image", "")),
                "image_path": str(row.get("image_path", "")),
                "local_image_path": str(image_path) if image_path else "",
                "link": str(row.get("link", ""))
            }

    def get_image_path(self, image_path, product_id):

        image_path = str(image_path).strip().replace("\\", "/")

        if image_path:

            path = Path(image_path)

            if path.is_absolute() and path.exists():
                return path

            parts = path.parts

            if parts and parts[0].lower() == "images":
                path = Path(*parts[1:])

            possible_path = self.image_dir / path

            if possible_path.exists():
                return possible_path
        for extension in [".jpg", ".jpeg", ".png", ".webp"]:

            possible_path = self.image_dir / f"{product_id}{extension}"

            if possible_path.exists():
                return possible_path

        return None

    def get_signature(self):
        signature_data = []
        for product_id, product in self.products.items():

            image_path = product["local_image_path"]

            if not image_path:
                continue

            path = Path(image_path)

            try:
                stat = path.stat()

                signature_data.append(
                    f"{product_id}_{stat.st_size}_{stat.st_mtime_ns}"
                )
            except OSError:
                continue

        signature = "|".join(signature_data)

        return hashlib.md5(signature.encode()).hexdigest()

    def load_or_create_index(self):

        signature = self.get_signature()

        if (
            self.index_path.exists()
            and self.ids_path.exists()
            and self.metadata_path.exists()
        ):

            with open(self.metadata_path, "r") as file:
                metadata = json.load(file)

            if (
                metadata.get("signature") == signature
                and metadata.get("model") == self.model_name
            ):

                self.index = faiss.read_index(str(self.index_path))

                with open(self.ids_path, "r") as file:
                    self.product_ids = json.load(file)

                if self.index.ntotal == len(self.product_ids):
                    print("Loaded saved image embeddings.")
                    return
                
        self.create_embeddings(signature)

    def create_embeddings(self, signature):

        images = []
        product_ids = []

        for product_id, product in self.products.items():

            image_path = product["local_image_path"]

            if not image_path:
                continue

            try:
                with Image.open(image_path) as img:
                    images.append(img.convert("RGB").copy())
                    product_ids.append(product_id)

            except Exception:
                continue

        if not images:
            raise ValueError("No valid product images found.")

        embeddings_list = []

        # Process images in batches to control memory usage
        batch_size = 32

        for start in range(0, len(images), batch_size):

            batch = images[start:start + batch_size]

            embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            embeddings_list.append(embeddings.astype("float32"))

        embeddings = np.vstack(embeddings_list)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.product_ids = product_ids

        faiss.write_index(self.index, str(self.index_path))

        with open(self.ids_path, "w") as file:
            json.dump(self.product_ids, file)

        metadata = {
            "signature": signature,
            "model": self.model_name
        }

        with open(self.metadata_path, "w") as file:
            json.dump(metadata, file)

        print("Image embeddings saved successfully.")

    def search(self, image_file, k=5):

        if isinstance(image_file, (bytes, bytearray)):
            image_file = io.BytesIO(image_file)

        with Image.open(image_file) as img:
            query_image = img.convert("RGB").copy()

        query_embedding = self.model.encode(
            [query_image],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")
        k = min(k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for rank, idx in enumerate(indices[0], start=1):

            if idx == -1:
                continue

            product_id = self.product_ids[idx]

            product = self.products[product_id].copy()

            product["similarity"] = round(
                float(scores[0][rank - 1]) * 100, 2
            )

            product["rank"] = rank

            results.append(product)

        return results