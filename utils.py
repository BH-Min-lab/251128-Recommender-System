# utils.py
# Utility functions for XGBoost GPU Learning to Rank v26

import os
import logging
import numpy as np
import pandas as pd

from config import OUT_DIR, LOG_PATH


def setup_logging():
    """Setup console + file logging"""
    os.makedirs(OUT_DIR, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Format
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # File handler
    fh = logging.FileHandler(LOG_PATH, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def safe_mkdir(path):
    """Create directory if not exists"""
    os.makedirs(path, exist_ok=True)


def price_bucket(price):
    """Categorize price into buckets"""
    if pd.isna(price):
        return -1
    if price <= 50:
        return 0
    if price <= 100:
        return 1
    if price <= 200:
        return 2
    if price <= 500:
        return 3
    return 4


def get_price_bonus(price):
    """Calculate price bonus for v5 heuristic"""
    if pd.isna(price):
        return 1.0
    if price <= 50:
        return 1.5
    elif price <= 500:
        return 1.2 if price > 200 else 1.0
    else:
        return 0.8


def split_users_hash(users, valid_ratio=0.2, seed=42):
    """
    Split users into train/val sets using hash-based splitting
    Ensures reproducibility and no data leakage
    """
    users = np.array(users)
    h = pd.util.hash_pandas_object(pd.Series(users), index=False).values
    rng = np.random.RandomState(seed)
    h = h ^ rng.randint(0, 2**32 - 1, size=h.shape, dtype=np.uint64)
    mask_val = (h % 100) < int(valid_ratio * 100)
    train_users = users[~mask_val].tolist()
    val_users = users[mask_val].tolist()
    return train_users, val_users


def filter_empty_groups(X, pairs, y, groups, users):
    """Remove users with empty groups from training data"""
    keep_user_mask = [g > 0 for g in groups]
    kept_users = [u for u, k in zip(users, keep_user_mask) if k]
    m = pairs["user_id"].isin(set(kept_users))
    X2 = X[m].reset_index(drop=True)
    pairs2 = pairs[m].reset_index(drop=True)
    y2 = y[m.values]
    grp = pairs2.groupby("user_id").size()
    groups2 = [int(grp.get(u, 0)) for u in kept_users]
    return X2, pairs2, y2, groups2, kept_users


def make_weighted_label(pairs, future_label_map):
    """Create weighted labels from future events"""
    return np.array(
        [future_label_map.get((u, i), 0) for u, i in pairs[["user_id", "item_id"]].itertuples(index=False)],
        dtype=np.float32,
    )
