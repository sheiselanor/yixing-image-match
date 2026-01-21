import os
from pathlib import Path

# -------------------- Image roots (3 server folders) --------------------
DEFAULT_ROOTS = ",".join(
    [
        "/var/www/html/YXPM/product/product",
        "/var/www/html/YXPM/bulk_product",
        "/var/www/html/YXPM/product_variation",
    ]
)

IMAGES_ROOTS = [
    Path(p.strip()).resolve()
    for p in os.environ.get("IMAGES_ROOTS", DEFAULT_ROOTS).split(",")
    if p.strip()
]

# -------------------- Index paths --------------------
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "./index")).resolve()
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = INDEX_DIR / "vx.index"
META_PATH = INDEX_DIR / "vx.meta.json"
EMB_PATH = INDEX_DIR / "vx.embeddings.npy"

# -------------------- Public URL prefixes --------------------
PUBLIC_PREFIX_PRODUCT = os.environ.get(
    "PUBLIC_PREFIX_PRODUCT",
    "https://yixinggoldsmith.com/storage/product",
)
PUBLIC_PREFIX_BULK = os.environ.get(
    "PUBLIC_PREFIX_BULK",
    "https://yixinggoldsmith.com/storage/bulk_product",
)
PUBLIC_PREFIX_VARIATION = os.environ.get(
    "PUBLIC_PREFIX_VARIATION",
    "https://yixinggoldsmith.com/storage/product_variation",
)

# -------------------- Model & thresholds --------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "clip-ViT-B-32")
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.18"))

# -------------------- Admin --------------------
ADMIN_KEY = os.environ.get("ADMIN_KEY", "dev-key")

# -------------------- Search knobs --------------------
HNSW_EF = int(os.environ.get("HNSW_EF", "256"))
TOP_K_INTERNAL = int(os.environ.get("TOP_K_INTERNAL", "400"))

# -------------------- Scheduler (UTC-based, but set Malaysia time) --------------------
# We run the scheduler in UTC to avoid timezone errors.
# You configure the target Malaysia time in REINDEX_TIME_MY (default 05:00).
ENABLE_SCHEDULER = os.environ.get("ENABLE_SCHEDULER", "false").lower() == "true"

REINDEX_TIME_MY = os.environ.get("REINDEX_TIME_MY", "05:00")  # Malaysia time HH:MM
MY_UTC_OFFSET_HOURS = int(os.environ.get("MY_UTC_OFFSET_HOURS", "8"))  # Malaysia UTC+8

SCHEDULER_TIMEZONE = os.environ.get("SCHEDULER_TIMEZONE", "UTC")
