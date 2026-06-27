"""Build chroma_multimodal index (M2L1 logic) for M2L2 lab."""
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

PROJECT = Path(__file__).resolve().parent
IMG_DIR = PROJECT / "recipe_images"
DB_DIR = Path.home() / "chroma_multimodal"

with open(PROJECT / "structured_restaurant_data.json") as f:
    restaurants = json.load(f)
with open(PROJECT / "augmented_food_recipe.json") as f:
    recipes = json.load(f)

image_paths = sorted(glob.glob(str(IMG_DIR / "**" / "*.png"), recursive=True))
print(f"Loaded restaurants: {len(restaurants)}, recipes: {len(recipes)}, images: {len(image_paths)}")

text_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts, batch_size=64):
    return text_model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)

device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()

@torch.no_grad()
def embed_images(paths, batch_size=16):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

article_docs = []
for i, r in enumerate(restaurants):
    name = str(r.get("name", "")).strip()
    if not name:
        continue
    text = (
        f"Restaurant: {name}\n"
        f"Cuisine: {r.get('food_style', '')}\n"
        f"Location: {r.get('location', '')}"
    )
    article_docs.append(
        Document(
            page_content=text.strip(),
            metadata={
                "doc_id": f"rest_{i}",
                "cuisine": r.get("food_style"),
                "location": r.get("location"),
                "source": "restaurant",
            },
        )
    )

image_docs = []
for i, (p, rec) in enumerate(zip(image_paths, recipes)):
    image_docs.append(
        Document(
            page_content=rec.get("name", f"recipe image {i}"),
            metadata={
                "doc_id": f"img_{i}",
                "image_path": p,
                "source": "recipe_image",
                "recipe_id": rec.get("id"),
                "cuisine": rec.get("cuisine"),
            },
        )
    )

if DB_DIR.is_dir():
    shutil.rmtree(DB_DIR)

db_path = str(DB_DIR.resolve())
A = embed_texts([d.page_content for d in article_docs])
article_db = Chroma(collection_name="restaurant_articles", persist_directory=db_path)
article_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in article_docs],
    embeddings=A.tolist(),
    documents=[d.page_content for d in article_docs],
    metadatas=[d.metadata for d in article_docs],
)
print("Article DB ready")

V = embed_images([d.metadata["image_path"] for d in image_docs])
image_db = Chroma(collection_name="food_images", persist_directory=db_path)
image_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in image_docs],
    embeddings=V.tolist(),
    documents=[d.page_content for d in image_docs],
    metadatas=[d.metadata for d in image_docs],
)
print("Image DB ready")
print(f"Multimodal Vector Index COMPLETE at {db_path}")
