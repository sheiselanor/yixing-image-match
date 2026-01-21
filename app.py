import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import hnswlib
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sentence_transformers import SentenceTransformer

from config import (
    ADMIN_KEY,
    EMB_PATH,
    ENABLE_SCHEDULER,
    HNSW_EF,
    INDEX_PATH,
    META_PATH,
    MIN_SCORE,
    MODEL_NAME,
    MY_UTC_OFFSET_HOURS,
    PUBLIC_PREFIX_BULK,
    PUBLIC_PREFIX_PRODUCT,
    PUBLIC_PREFIX_VARIATION,
    REINDEX_TIME_MY,
    SCHEDULER_TIMEZONE,
    TOP_K_INTERNAL,
)
from indexer import build_index

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


app = FastAPI(title="YI XING Visual Search", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
index = None
vecs = None

meta_paths: List[str] = []
meta_buckets: List[str] = []
meta_filenames: List[str] = []
embed_dim: int = 512


def load_model() -> SentenceTransformer:
    global model
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    return model


def load_assets() -> bool:
    global index, vecs, meta_paths, meta_buckets, meta_filenames, embed_dim

    if not (INDEX_PATH.exists() and META_PATH.exists() and EMB_PATH.exists()):
        return False

    meta = json.loads(META_PATH.read_text())
    meta_paths = meta["paths"]
    meta_buckets = meta["buckets"]
    meta_filenames = meta["filenames"]
    embed_dim = int(meta.get("dim", 512))

    idx = hnswlib.Index(space="cosine", dim=embed_dim)
    idx.load_index(str(INDEX_PATH))
    idx.set_ef(HNSW_EF)

    vectors = np.load(str(EMB_PATH)).astype("float32")

    index = idx
    vecs = vectors
    return True


def ensure_loaded() -> None:
    if globals().get("index") is None or globals().get("vecs") is None:
        if not load_assets():
            raise HTTPException(
                status_code=503,
                detail="Index not ready. Call /reindex first.",
            )


def to_public_url(bucket: str, filename: str) -> str:
    if bucket == "bulk_product":
        return f"{PUBLIC_PREFIX_BULK.rstrip('/')}/{filename}"
    if bucket == "product_variation":
        return f"{PUBLIC_PREFIX_VARIATION.rstrip('/')}/{filename}"
    return f"{PUBLIC_PREFIX_PRODUCT.rstrip('/')}/{filename}"


# ---------------- Query preprocessing (WhatsApp robustness) ----------------
def limit_size(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.BICUBIC)


def normalize_img(img: Image.Image) -> Image.Image:
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    img = ImageEnhance.Brightness(img).enhance(1.08)
    return img.filter(
        ImageFilter.UnsharpMask(radius=1.0, percent=140, threshold=3)
    )


def foreground_crop(img: Image.Image, pad: int = 12) -> Image.Image:
    w, h = img.size
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    arr = np.array(edges)
    thr = max(18, int(arr.mean() + arr.std() * 0.6))

    ys, xs = np.where(arr > thr)
    if xs.size > 200 and ys.size > 200:
        x1 = max(0, int(xs.min()) - pad)
        x2 = min(w - 1, int(xs.max()) + pad)
        y1 = max(0, int(ys.min()) - pad)
        y2 = min(h - 1, int(ys.max()) + pad)

        if (x2 - x1) > w * 0.25 and (y2 - y1) > h * 0.25:
            return img.crop((x1, y1, x2, y2))

    return img


def center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    x1 = (w - s) // 2
    y1 = (h - s) // 2
    return img.crop((x1, y1, x1 + s, y1 + s))


def embed_query(data: bytes) -> np.ndarray:
    img0 = Image.open(BytesIO(data))
    img0 = ImageOps.exif_transpose(img0).convert("RGB")
    img0 = limit_size(img0, 1024)

    m = load_model()

    variants = [
        normalize_img(img0),
        normalize_img(foreground_crop(img0)),
        normalize_img(center_square(img0)),
    ]

    embs = m.encode(
        variants,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    q = np.mean(embs, axis=0, keepdims=True)
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    return q.astype("float32")


# ---------------- ORB tie-break for tiny details ----------------
def orb_score(query_rgb: np.ndarray, cand_rgb: np.ndarray) -> float:
    if cv2 is None:
        return 0.0

    try:
        q_gray = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY)
        c_gray = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=800)
        _kq, dq = orb.detectAndCompute(q_gray, None)
        _kc, dc = orb.detectAndCompute(c_gray, None)
        if dq is None or dc is None:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(dq, dc)
        if not matches:
            return 0.0

        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:40]
        avg_dist = float(np.mean([m.distance for m in good]))
        return 1.0 / (1.0 + avg_dist)

    except Exception:
        return 0.0


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/config")
def config():
    return {
        "MODEL_NAME": MODEL_NAME,
        "MIN_SCORE": MIN_SCORE,
        "HNSW_EF": HNSW_EF,
        "TOP_K_INTERNAL": TOP_K_INTERNAL,
        "index_loaded": (
            INDEX_PATH.exists() and META_PATH.exists() and EMB_PATH.exists()
        ),
        "orb_enabled": cv2 is not None,
        "scheduler_timezone": SCHEDULER_TIMEZONE,
        "reindex_time_my": REINDEX_TIME_MY,
    }


@app.post("/reindex")
def reindex(
    x_admin_key: str = Header(default=""),
    admin_key: str = Header(default=""),
    key: str = Query(default=""),
):
    supplied = x_admin_key or admin_key or key
    if supplied != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    lock = Path("/tmp/vx_reindex.lock")
    if lock.exists():
        raise HTTPException(status_code=409, detail="Reindex already running")

    try:
        lock.write_text(str(time.time()))
        build_index()
        ok = load_assets()
        return {"ok": ok, "message": "Index rebuilt and loaded"}
    finally:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/search")
async def search(
    file: UploadFile = File(...),
    top_k_products: int = 3,
    k_internal: int = 0,
):
    ensure_loaded()

    data = await file.read()
    q = embed_query(data)

    k = k_internal if k_internal > 0 else TOP_K_INTERNAL
    labels, _distances = index.knn_query(q, k=k)

    cand_idx = labels[0]
    cand_idx = cand_idx[cand_idx >= 0]
    if cand_idx.size == 0:
        return JSONResponse({"matches": [], "low_confidence": True})

    cand_vecs = vecs[cand_idx]
    sims = (cand_vecs @ q[0]).astype(float)

    order = np.argsort(-sims)
    cand_idx = cand_idx[order]
    sims = sims[order]

    # ORB tie-break among close candidates (top 20)
    if cv2 is not None and len(cand_idx) >= 5:
        close_gap = sims[0] - sims[min(4, len(sims) - 1)]
        if close_gap < 0.03:
            q_img = Image.open(BytesIO(data))
            q_img = ImageOps.exif_transpose(q_img).convert("RGB")
            q_np = np.array(normalize_img(limit_size(q_img, 800)))

            rerank_n = min(20, len(cand_idx))
            orb_scores = []
            for i in range(rerank_n):
                p = meta_paths[int(cand_idx[i])]
                try:
                    c_img = Image.open(p).convert("RGB")
                    c_np = np.array(normalize_img(limit_size(c_img, 800)))
                    orb_scores.append(orb_score(q_np, c_np))
                except Exception:
                    orb_scores.append(0.0)

            orb_scores = np.array(orb_scores, dtype=float)
            combined = sims[:rerank_n] + (0.12 * orb_scores)
            new_order = np.argsort(-combined)

            cand_idx[:rerank_n] = cand_idx[:rerank_n][new_order]
            sims[:rerank_n] = sims[:rerank_n][new_order]

    matches = []
    limit = min(top_k_products, len(cand_idx))
    for i in range(limit):
        ix = int(cand_idx[i])
        score = float(sims[i])
        bucket = meta_buckets[ix]
        filename = meta_filenames[ix]

        matches.append(
            {
                "bucket": bucket,
                "filename": filename,
                "score": round(score, 4),
                "image_url": to_public_url(bucket, filename),
                "image_path": meta_paths[ix],
            }
        )

    low_conf = matches[0]["score"] < max(0.22, MIN_SCORE)
    return JSONResponse({"matches": matches, "low_confidence": low_conf})


# ---------------- Scheduler: Set MY time, run UTC ----------------
def malaysia_time_to_utc_hm(time_my: str, offset_hours: int) -> Tuple[int, int]:
    # time_my is HH:MM in Malaysia time (UTC+offset_hours)
    hh_str, mm_str = time_my.split(":")
    hh_my = int(hh_str)
    mm = int(mm_str)
    hh_utc = (hh_my - offset_hours) % 24
    return hh_utc, mm


def start_scheduler() -> None:
    if not ENABLE_SCHEDULER:
        return

    from apscheduler.schedulers.background import BackgroundScheduler

    hh_utc, mm_utc = malaysia_time_to_utc_hm(REINDEX_TIME_MY, MY_UTC_OFFSET_HOURS)

    scheduler = BackgroundScheduler(timezone=SCHEDULER_TIMEZONE)

    def job():
        lock = Path("/tmp/vx_reindex.lock")
        if lock.exists():
            return

        try:
            lock.write_text(str(time.time()))
            build_index()
            load_assets()
        finally:
            try:
                lock.unlink(missing_ok=True)
            except Exception:
                pass

    scheduler.add_job(job, "cron", hour=hh_utc, minute=mm_utc)
    scheduler.start()
    print(
        "[SCHED] Daily reindex at Malaysia",
        REINDEX_TIME_MY,
        "=> UTC",
        f"{hh_utc:02d}:{mm_utc:02d}",
        "timezone used:",
        SCHEDULER_TIMEZONE,
    )


@app.on_event("startup")
def on_startup():
    load_assets()
    start_scheduler()
