# item_profile.py
# Item behavior profiling and clustering for XGBoost GPU Learning to Rank v26

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import N_CLUSTERS


def build_item_behavior_profile(df):
    """
    Build item behavior profiles from user interaction data
    """
    logging.info("=" * 50)
    logging.info("[ItemProfile] Building item behavior profiles...")
    logging.info(f"  Input data: {len(df):,} rows, {df['item_id'].nunique():,} unique items")
    logging.info(f"  Event distribution: {df['event_type'].value_counts().to_dict()}")

    # 1. Basic event counts
    event_counts = df.groupby(["item_id", "event_type"]).size().unstack(fill_value=0)
    event_counts.columns = [f"cnt_{c}" for c in event_counts.columns]
    event_counts = event_counts.reset_index()

    for col in ["cnt_view", "cnt_cart", "cnt_purchase"]:
        if col not in event_counts.columns:
            event_counts[col] = 0

    # 2. Average views per user (per item)
    view_df = df[df["event_type"] == "view"]
    user_view_per_item = view_df.groupby(["item_id", "user_id"]).size().reset_index(name="views")
    avg_views_per_user = user_view_per_item.groupby("item_id")["views"].mean().reset_index()
    avg_views_per_user.columns = ["item_id", "avg_views_per_user"]

    # 3. Unique viewers
    unique_viewers = view_df.groupby("item_id")["user_id"].nunique().reset_index()
    unique_viewers.columns = ["item_id", "unique_viewers"]

    # 4. View time range
    view_time_range = view_df.groupby("item_id")["event_time"].agg(["min", "max"]).reset_index()
    view_time_range["view_span_hours"] = (
        view_time_range["max"] - view_time_range["min"]
    ).dt.total_seconds() / 3600.0
    view_time_range = view_time_range[["item_id", "view_span_hours"]]

    # 5. Session-based views
    if "user_session" in df.columns:
        session_views = view_df.groupby(["item_id", "user_session"]).size().reset_index(name="session_views")
        avg_session_views = session_views.groupby("item_id")["session_views"].mean().reset_index()
        avg_session_views.columns = ["item_id", "avg_session_views"]
        unique_sessions = session_views.groupby("item_id")["user_session"].nunique().reset_index()
        unique_sessions.columns = ["item_id", "unique_sessions"]
    else:
        avg_session_views = pd.DataFrame({"item_id": event_counts["item_id"], "avg_session_views": 1.0})
        unique_sessions = pd.DataFrame({"item_id": event_counts["item_id"], "unique_sessions": 1})

    # 6. Price info
    item_price = df.groupby("item_id")["price"].mean().reset_index()
    item_price.columns = ["item_id", "avg_price"]

    # 7. Brand info (mode)
    if "brand" in df.columns:
        item_brand = df.groupby("item_id")["brand"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown").reset_index()
        item_brand.columns = ["item_id", "brand"]
    else:
        item_brand = pd.DataFrame({"item_id": event_counts["item_id"], "brand": "unknown"})

    # 7-1. real_category info (LLM-generated category)
    if "real_category" in df.columns:
        item_category = df.groupby("item_id")["real_category"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "other").reset_index()
        item_category.columns = ["item_id", "real_category"]
    else:
        item_category = pd.DataFrame({"item_id": event_counts["item_id"], "real_category": "other"})

    # 8. Peak hour/day of week
    view_df = view_df.copy()
    view_df["hour"] = view_df["event_time"].dt.hour
    view_df["dow"] = view_df["event_time"].dt.dayofweek

    peak_hour = view_df.groupby("item_id")["hour"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12).reset_index()
    peak_hour.columns = ["item_id", "peak_hour"]

    peak_dow = view_df.groupby("item_id")["dow"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 3).reset_index()
    peak_dow.columns = ["item_id", "peak_dow"]

    # Merge all
    profile = event_counts.copy()
    profile = profile.merge(avg_views_per_user, on="item_id", how="left")
    profile = profile.merge(unique_viewers, on="item_id", how="left")
    profile = profile.merge(view_time_range, on="item_id", how="left")
    profile = profile.merge(avg_session_views, on="item_id", how="left")
    profile = profile.merge(unique_sessions, on="item_id", how="left")
    profile = profile.merge(item_price, on="item_id", how="left")
    profile = profile.merge(item_brand, on="item_id", how="left")
    profile = profile.merge(item_category, on="item_id", how="left")
    profile = profile.merge(peak_hour, on="item_id", how="left")
    profile = profile.merge(peak_dow, on="item_id", how="left")

    # Conversion rate calculation (with smoothing)
    profile["view_to_cart_rate"] = (profile["cnt_cart"] + 1) / (profile["cnt_view"] + 100)
    profile["cart_to_purchase_rate"] = (profile["cnt_purchase"] + 1) / (profile["cnt_cart"] + 10)
    profile["view_to_purchase_rate"] = (profile["cnt_purchase"] + 1) / (profile["cnt_view"] + 100)

    # Fill missing values
    profile["avg_views_per_user"] = profile["avg_views_per_user"].fillna(1.0)
    profile["unique_viewers"] = profile["unique_viewers"].fillna(1)
    profile["view_span_hours"] = profile["view_span_hours"].fillna(0)
    profile["avg_session_views"] = profile["avg_session_views"].fillna(1.0)
    profile["unique_sessions"] = profile["unique_sessions"].fillna(1)
    profile["avg_price"] = profile["avg_price"].fillna(profile["avg_price"].median())
    profile["peak_hour"] = profile["peak_hour"].fillna(12)
    profile["peak_dow"] = profile["peak_dow"].fillna(3)

    # Detailed profile statistics logging
    logging.info(f"  Item profiles created: {len(profile):,} items")
    logging.info(f"  [Profile Stats]")
    logging.info(f"    avg_views_per_user: min={profile['avg_views_per_user'].min():.2f}, max={profile['avg_views_per_user'].max():.2f}, mean={profile['avg_views_per_user'].mean():.2f}")
    logging.info(f"    avg_price: min={profile['avg_price'].min():.2f}, max={profile['avg_price'].max():.2f}, mean={profile['avg_price'].mean():.2f}")

    n_brands = profile["brand"].nunique()
    logging.info(f"    unique_brands: {n_brands}")

    profile["real_category"] = profile["real_category"].fillna("other")
    n_categories = profile["real_category"].nunique()
    logging.info(f"    unique_categories: {n_categories}")

    return profile


def cluster_items_by_behavior(item_profile, n_clusters=N_CLUSTERS):
    """
    K-Means clustering based on item behavior profiles (including price)
    """
    logging.info("=" * 50)
    logging.info(f"[Cluster] Clustering items into {n_clusters} groups...")

    cluster_features = [
        "avg_views_per_user",
        "view_span_hours",
        "avg_session_views",
        "view_to_cart_rate",
        "view_to_purchase_rate",
        "avg_price",
        "peak_hour",
    ]
    logging.info(f"  Clustering features: {cluster_features}")

    X = item_profile[cluster_features].copy()

    # Log transform (normalize skewed distributions)
    X["avg_views_per_user"] = np.log1p(X["avg_views_per_user"])
    X["view_span_hours"] = np.log1p(X["view_span_hours"])
    X["avg_session_views"] = np.log1p(X["avg_session_views"])
    X["avg_price"] = np.log1p(X["avg_price"])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means clustering
    logging.info(f"  Running K-Means (k={n_clusters}, n_init=10)...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    logging.info(f"  K-Means inertia: {kmeans.inertia_:.2f}")

    item_profile = item_profile.copy()
    item_profile["cluster_id"] = clusters

    # NEW v26: Create fine_category (real_category + cluster_id)
    item_profile["fine_category"] = (
        item_profile["real_category"].astype(str) + "_c" + item_profile["cluster_id"].astype(str)
    )
    n_fine_categories = item_profile["fine_category"].nunique()
    logging.info(f"  NEW: fine_category created: {n_fine_categories} unique fine categories")

    # Cluster statistics logging
    cluster_stats = item_profile.groupby("cluster_id").agg({
        "item_id": "count",
        "avg_price": "mean",
        "avg_views_per_user": "mean",
        "view_to_purchase_rate": "mean",
        "cnt_purchase": "sum"
    }).reset_index()
    cluster_stats.columns = ["cluster_id", "n_items", "avg_price", "avg_views_per_user", "avg_conversion", "total_purchases"]

    logging.info(f"  [Cluster Distribution]")
    logging.info(f"    Items per cluster: min={cluster_stats['n_items'].min()}, max={cluster_stats['n_items'].max()}, avg={cluster_stats['n_items'].mean():.1f}")

    # Price tier distribution per cluster
    price_bins = [0, 50, 100, 200, 500, float('inf')]
    price_labels = ['<50', '50-100', '100-200', '200-500', '>500']
    item_profile['price_tier'] = pd.cut(item_profile['avg_price'], bins=price_bins, labels=price_labels)

    return item_profile, cluster_stats, kmeans, scaler


def build_cluster_top_items(item_profile, item_stats, top_k=20):
    """
    Build top items per cluster (based on purchase conversion + popularity)
    """
    logging.info("=" * 50)
    logging.info("[ClusterTop] Building top items per cluster...")

    merged = item_profile.merge(
        item_stats[["item_id", "item_purchase_rate", "item_view_pop"]],
        on="item_id",
        how="left"
    )
    merged["item_purchase_rate"] = merged["item_purchase_rate"].fillna(0)
    merged["item_view_pop"] = merged["item_view_pop"].fillna(0)

    # Score: conversion rate * (1 + log(popularity))
    merged["cluster_score"] = (
        merged["item_purchase_rate"] * (1 + np.log1p(merged["item_view_pop"]))
    )

    cluster_top_items = {}
    for cluster_id in merged["cluster_id"].unique():
        cluster_items = merged[merged["cluster_id"] == cluster_id]
        scored_items = cluster_items[cluster_items["cluster_score"] > 0]

        if len(scored_items) >= top_k:
            top_items = scored_items.nlargest(top_k, "cluster_score")["item_id"].tolist()
        elif len(scored_items) > 0:
            top_items = scored_items.nlargest(len(scored_items), "cluster_score")["item_id"].tolist()
            remaining = top_k - len(top_items)
            filler_items = cluster_items[~cluster_items["item_id"].isin(top_items)].nlargest(remaining, "item_view_pop")["item_id"].tolist()
            top_items.extend(filler_items)
        else:
            top_items = cluster_items.nlargest(top_k, "item_view_pop")["item_id"].tolist()

        cluster_top_items[cluster_id] = top_items

    logging.info(f"  Built top-{top_k} items for {len(cluster_top_items)} clusters")
    return cluster_top_items


def get_user_cluster_affinity(user_id, user_view_history, item_profile):
    """
    Calculate user's cluster distribution from viewed items
    """
    if user_view_history is None or len(user_view_history) == 0:
        return {}

    viewed_items = set(user_view_history)
    viewed_clusters = item_profile[item_profile["item_id"].isin(viewed_items)]["cluster_id"].value_counts()

    if len(viewed_clusters) == 0:
        return {}

    total = viewed_clusters.sum()
    affinity = {int(k): float(v / total) for k, v in viewed_clusters.items()}
    return affinity


def build_brand_cluster_map(item_profile):
    """
    Build brand to main cluster mapping
    """
    logging.info("=" * 50)
    logging.info("[BrandCluster] Building brand-cluster mapping...")

    brand_cluster = item_profile.groupby(["brand", "cluster_id"]).size().reset_index(name="count")

    brand_main_cluster = brand_cluster.loc[
        brand_cluster.groupby("brand")["count"].idxmax()
    ][["brand", "cluster_id"]]
    brand_main_cluster.columns = ["brand", "main_cluster_id"]

    brand_cluster_map = dict(zip(brand_main_cluster["brand"], brand_main_cluster["main_cluster_id"]))

    logging.info(f"  Mapped {len(brand_cluster_map)} brands to clusters")
    return brand_cluster_map


def build_item_cluster_map(item_profile):
    """Build item_id to (cluster_id, avg_views, conversion_rate) mapping"""
    item_cluster_map = {}
    for _, row in item_profile.iterrows():
        item_cluster_map[row["item_id"]] = (
            int(row["cluster_id"]),
            float(row["avg_views_per_user"]),
            float(row["view_to_purchase_rate"])
        )
    return item_cluster_map


def build_item_fine_category_map(item_profile):
    """Build item_id to fine_category mapping"""
    return item_profile.set_index("item_id")["fine_category"].to_dict()
