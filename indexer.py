import json
from pathlib import Path
from typing import List, Tuple

import hnswlib
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import EMB_PATH, IMAGES_ROOTS, INDEX_PATH, META_PATH, MODEL_NAME

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def bucket_from_root(root: Path) -> str:
    s = str(root).replace("\\", "/").lower()
    if s.endswith("/product_variation"):
        return "product_variation"
    if s.endswith("/bulk_product"):
        return "bulk_product"
    return "product"  # /product/product


def list_image_files(roots: List[Path]) -> Tuple[List[Path], List[str], List[str]]:
    paths: List[Path] = []
    buckets: List[str] = []
    filenames: List[str] = []

    for root in roots:
        bucket = bucket_from_root(root)
        if not root.exists():
            print(f"[WARN] Root not found: {root}")
            continue

        for fp in root.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTS:
                paths.append(fp.resolve())
                buckets.append(bucket)
                filenames.append(fp.name)  # filename used in public URL

    return paths, buckets, filenames


def embed_images(
    paths: List[Path],
    model: SentenceTransformer,
    batch_size: int = 16,
) -> np.ndarray:
    vecs = []
    batch = []

    for p in tqdm(paths, desc="Embedding images"):
        try:
            img = Image.open(p).convert("RGB")
            batch.append(img)
            if len(batch) >= batch_size:
                emb = model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype("float32")
                vecs.append(emb)
                batch = []
        except Exception as e:
            print(f"[WARN] Skip {p}: {e}")

    if batch:
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        vecs.append(emb)

    if not vecs:
        return np.zeros((0, 512), dtype="float32")

    return np.vstack(vecs)


def build_index() -> None:
    print("[INFO] Scanning roots:")
    for r in IMAGES_ROOTS:
        print(" -", r)

    paths, buckets, filenames = list_image_files(IMAGES_ROOTS)
    if not paths:
        raise RuntimeError("No images found under IMAGES_ROOTS.")

    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    vecs = embed_images(paths, model)
    if vecs.shape[0] == 0:
        raise RuntimeError("Embedding produced 0 vectors.")

    dim = vecs.shape[1]
    print(f"[INFO] Embedded {vecs.shape[0]} images; dim={dim}")

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=vecs.shape[0], ef_construction=200, M=16)
    index.add_items(vecs)
    index.save_index(str(INDEX_PATH))

    np.save(str(EMB_PATH), vecs)

    meta = {
        "paths": [str(p) for p in paths],
        "buckets": buckets,
        "filenames": filenames,
        "dim": int(dim),
        "model": MODEL_NAME,
    }
    META_PATH.write_text(json.dumps(meta))

    print(f"[OK] Wrote index: {INDEX_PATH}")
    print(f"[OK] Wrote embeddings: {EMB_PATH}")
    print(f"[OK] Wrote meta: {META_PATH}")


if __name__ == "__main__":
    build_index()
