# config.py
# Configuration settings for XGBoost GPU Learning to Rank v26

import os
from datetime import datetime

# =========================
# Paths
# =========================
DATA_DIR = "../data"
TRAIN_PATH = os.path.join(DATA_DIR, "new_train.parquet")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

OUT_DIR = "./output_Rec_011"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = os.path.join(OUT_DIR, f"output_xgb_gpu_rerank_v26_{TIMESTAMP}.csv")
LOG_PATH = os.path.join(OUT_DIR, f"xgb_gpu_rerank_v26_log_{TIMESTAMP}.txt")
ANALYSIS_PATH = os.path.join(OUT_DIR, f"xgb_gpu_rerank_v26_analysis_{TIMESTAMP}.json")


# =========================
# Candidate Generation Settings
# =========================
CAND_CART = 50
CAND_REPEAT = 200
CAND_RECENT = 200
CAND_POP = 200
CAND_MAX = 600  # Maximum candidates per user after deduplication


# =========================
# Training Settings
# =========================
VAL_FUTURE_DAYS = 10  # Days for label window
MAX_TRAIN_USERS = 60000  # Maximum training users


# =========================
# Clustering Settings
# =========================
N_CLUSTERS = 50  # Number of item behavior profile clusters


# =========================
# XGBoost GPU Parameters (Optimized for RTX 3090 24GB)
# =========================
XGB_GPU_PARAMS = {
    # GPU settings
    "device": "cuda",
    "tree_method": "hist",

    # Learning parameters
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",
    "learning_rate": 0.05,
    "n_estimators": 5000,

    # Tree structure
    "max_depth": 8,
    "max_leaves": 127,
    "min_child_weight": 50,

    # Regularization
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 0.1,

    # Subsampling
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.8,

    # GPU Memory optimization
    "max_bin": 256,

    # Random seed
    "seed": 42,

    # Verbose
    "verbosity": 1,
}

EARLY_STOPPING_ROUNDS = 100


# =========================
# Inference Settings
# =========================
BATCH_SIZE = 50000
RECENT_DAYS = 14
BLEND_ALPHA = 0.35
USE_RANK_BLEND = True
MIN_VIEWS_FOR_FULL_SCORE = 5


# =========================
# Event Weights for Implicit Labels
# =========================
EVENT_WEIGHTS = {
    "purchase": 5,
    "cart": 3,
    "view": 1,
}


# =========================
# Feature Columns
# =========================
FEATURE_COLS = [
    "ui_view_cnt", "ui_view_cnt_40h",
    "repeat2", "repeat3", "repeat5",
    "ui_cart_flag",
    "ui_last_hours_ago",
    "ui_last_dow", "ui_last_hour",
    "is_peak", "is_active", "is_thu", "is_fri", "is_sat",
    "item_view_pop", "item_purchase_pop", "item_purchase_rate",
    "item_price", "price_bucket", "price_bonus",
    "src_cart", "src_repeat", "src_recent", "src_popular",
    "src_priority", "v5_score",
    "item_cluster_id", "item_avg_views_per_user", "item_profile_conversion",
    "cluster_match_score",
    "fine_category_match_score",
]
