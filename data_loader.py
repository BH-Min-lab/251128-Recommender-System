# data_loader.py
# Data loading and global statistics for XGBoost GPU Learning to Rank v26

import logging
import numpy as np
import pandas as pd

from config import TRAIN_PATH, SAMPLE_PATH
from utils import price_bucket, get_price_bonus


def load_data():
    """Load training data and sample submission"""
    logging.info("[Load] reading parquet/csv...")
    df = pd.read_parquet(TRAIN_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    target_users = sample["user_id"].unique().tolist()

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"])

    logging.info(f"  rows={len(df):,}, target_users={len(target_users):,}")
    return df, sample, target_users


def build_global_stats(df):
    """Build global item statistics"""
    logging.info("[Global] building item stats...")
    view = df[df["event_type"] == "view"][["item_id"]]
    purchase = df[df["event_type"] == "purchase"][["item_id"]]

    item_view_pop = view["item_id"].value_counts()
    item_purchase_pop = purchase["item_id"].value_counts()

    item_view_pop = item_view_pop.rename("item_view_pop").reset_index().rename(columns={"index": "item_id"})
    item_purchase_pop = item_purchase_pop.rename("item_purchase_pop").reset_index().rename(columns={"index": "item_id"})

    item_price = df.groupby("item_id", as_index=False)["price"].mean().rename(columns={"price": "item_price"})

    stats = item_view_pop.merge(item_purchase_pop, on="item_id", how="left")
    stats = stats.merge(item_price, on="item_id", how="left")
    stats["item_purchase_pop"] = stats["item_purchase_pop"].fillna(0).astype(np.int64)

    stats["item_purchase_rate"] = (stats["item_purchase_pop"] + 1.0) / (stats["item_view_pop"] + 100.0)

    stats["price_bucket"] = stats["item_price"].apply(price_bucket).astype(np.int16)
    stats["price_bonus"] = stats["item_price"].apply(get_price_bonus).astype(np.float32)

    purchased_items = set(df[df["event_type"] == "purchase"]["item_id"].unique().tolist())
    purchased_view = stats[stats["item_id"].isin(purchased_items)].sort_values("item_view_pop", ascending=False)["item_id"].tolist()
    non_purchased_view = stats[~stats["item_id"].isin(purchased_items)].sort_values("item_view_pop", ascending=False)["item_id"].head(500).tolist()
    popular_items = purchased_view + non_purchased_view

    logging.info(f"  items with views={len(stats):,}, popular_pool={len(popular_items):,}")
    return stats, popular_items


def build_user_item_agg(df, recent_hours=40):
    """Build user-item interaction aggregates"""
    logging.info("[Agg] building user-item aggregates...")
    view = df[df["event_type"] == "view"][["user_id", "item_id", "event_time"]].copy()
    view["dow"] = view["event_time"].dt.dayofweek.astype(np.int8)
    view["hour"] = view["event_time"].dt.hour.astype(np.int8)

    max_time = view["event_time"].max()
    cutoff = max_time - pd.Timedelta(hours=recent_hours)

    ui_cnt = view.groupby(["user_id", "item_id"], as_index=False).size().rename(columns={"size": "ui_view_cnt"})

    view_sorted = view.sort_values("event_time")
    ui_last = view_sorted.drop_duplicates(["user_id", "item_id"], keep="last")[["user_id", "item_id", "event_time", "dow", "hour"]]
    ui_last = ui_last.rename(columns={"event_time": "ui_last_ts", "dow": "ui_last_dow", "hour": "ui_last_hour"})

    view_40 = view[view["event_time"] >= cutoff]
    ui_cnt_40 = view_40.groupby(["user_id", "item_id"], as_index=False).size().rename(columns={"size": "ui_view_cnt_40h"})

    ui = ui_cnt.merge(ui_last, on=["user_id", "item_id"], how="left")
    ui = ui.merge(ui_cnt_40, on=["user_id", "item_id"], how="left")
    ui["ui_view_cnt_40h"] = ui["ui_view_cnt_40h"].fillna(0).astype(np.int16)

    ui["ui_last_hours_ago"] = (max_time - ui["ui_last_ts"]).dt.total_seconds() / 3600.0
    ui["ui_last_hours_ago"] = ui["ui_last_hours_ago"].fillna(9999.0).astype(np.float32)

    ui["is_peak"] = ((ui["ui_last_hour"] >= 14) & (ui["ui_last_hour"] <= 17)).astype(np.int8)
    ui["is_active"] = ((ui["ui_last_hour"] >= 10) & (ui["ui_last_hour"] <= 18)).astype(np.int8)
    ui["is_thu"] = (ui["ui_last_dow"] == 3).astype(np.int8)
    ui["is_fri"] = (ui["ui_last_dow"] == 4).astype(np.int8)
    ui["is_sat"] = (ui["ui_last_dow"] == 5).astype(np.int8)

    cart = df[df["event_type"] == "cart"][["user_id", "item_id", "event_time"]].copy()
    if len(cart) > 0:
        cart_sorted = cart.sort_values("event_time")
        ui_cart = cart_sorted.drop_duplicates(["user_id", "item_id"], keep="last")[["user_id", "item_id"]]
        ui_cart["ui_cart_flag"] = 1
        ui = ui.merge(ui_cart, on=["user_id", "item_id"], how="left")
        ui["ui_cart_flag"] = ui["ui_cart_flag"].fillna(0).astype(np.int8)
    else:
        ui["ui_cart_flag"] = 0

    logging.info(f"  ui rows={len(ui):,}, max_time={max_time}")
    return ui, max_time


def prepare_indexed_lookups(ui, item_stats, purchased_pairs=None):
    """Create MultiIndex-based lookup objects for fast feature generation"""
    logging.info("  Preparing indexed lookups (MultiIndex).")

    ui_cols = [
        "ui_view_cnt", "ui_view_cnt_40h", "ui_cart_flag",
        "ui_last_hours_ago", "ui_last_dow", "ui_last_hour",
        "is_peak", "is_active", "is_thu", "is_fri", "is_sat",
    ]
    ui_cols = [c for c in ui_cols if c in ui.columns]

    ui_mi = ui.set_index(["user_id", "item_id"])[ui_cols]

    item_cols = [
        "item_view_pop", "item_purchase_pop", "item_purchase_rate",
        "item_price", "price_bucket", "price_bonus",
    ]
    item_cols = [c for c in item_cols if c in item_stats.columns]

    item_mi = item_stats.set_index("item_id")[item_cols]

    purchased_mi = None
    if purchased_pairs is not None and len(purchased_pairs) > 0:
        purchased_mi = pd.MultiIndex.from_frame(purchased_pairs[["user_id", "item_id"]])

    logging.info(f"  UI index: {len(ui_mi):,}, Item index: {len(item_mi):,}")
    return ui_mi, item_mi, purchased_mi
