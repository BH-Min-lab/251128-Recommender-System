# feature_engineering.py
# Feature engineering for XGBoost GPU Learning to Rank v26

import logging
import numpy as np
import pandas as pd
from collections import defaultdict

from config import FEATURE_COLS, MIN_VIEWS_FOR_FULL_SCORE


def build_feature_df(cand_dict, src_dict, ui, item_stats, max_time, exclude_purchased_pairs=None,
                     item_profile=None, user_view_map=None):
    """
    Build feature DataFrame for (user, item) candidate pairs
    Used for training data preparation
    """
    rows = []
    src_codes = []
    groups = []
    users = list(cand_dict.keys())

    for u in users:
        items = cand_dict[u]
        srcs = src_dict[u]
        groups.append(len(items))
        rows.extend([(u, i) for i in items])
        src_codes.extend(srcs)

    cand_df = pd.DataFrame(rows, columns=["user_id", "item_id"])
    cand_df["src_code"] = np.array(src_codes, dtype=np.int8)

    if exclude_purchased_pairs is not None and len(exclude_purchased_pairs) > 0:
        cand_df = cand_df.merge(exclude_purchased_pairs.assign(_p=1), on=["user_id", "item_id"], how="left")
        cand_df = cand_df[cand_df["_p"].isna()].drop(columns=["_p"])
        grp = cand_df.groupby("user_id").size()
        groups = [int(grp.get(u, 0)) for u in users]

    feats = cand_df.merge(ui, on=["user_id", "item_id"], how="left")

    feats["ui_view_cnt"] = feats["ui_view_cnt"].fillna(0).astype(np.int16)
    feats["ui_view_cnt_40h"] = feats["ui_view_cnt_40h"].fillna(0).astype(np.int16)
    feats["ui_cart_flag"] = feats["ui_cart_flag"].fillna(0).astype(np.int8)
    feats["ui_last_hours_ago"] = feats["ui_last_hours_ago"].fillna(9999.0).astype(np.float32)
    feats["ui_last_dow"] = feats["ui_last_dow"].fillna(-1).astype(np.int8)
    feats["ui_last_hour"] = feats["ui_last_hour"].fillna(-1).astype(np.int8)

    feats["repeat2"] = (feats["ui_view_cnt"] >= 2).astype(np.int8)
    feats["repeat3"] = (feats["ui_view_cnt"] >= 3).astype(np.int8)
    feats["repeat5"] = (feats["ui_view_cnt"] >= 5).astype(np.int8)

    feats = feats.merge(item_stats, on="item_id", how="left")
    feats["item_view_pop"] = feats["item_view_pop"].fillna(0).astype(np.int32)
    feats["item_purchase_pop"] = feats["item_purchase_pop"].fillna(0).astype(np.int32)
    feats["item_purchase_rate"] = feats["item_purchase_rate"].fillna(0.0).astype(np.float32)
    feats["item_price"] = feats["item_price"].fillna(-1.0).astype(np.float32)
    feats["price_bucket"] = feats["price_bucket"].fillna(-1).astype(np.int16)
    feats["price_bonus"] = feats["price_bonus"].fillna(1.0).astype(np.float32)

    for c in ["is_peak", "is_active", "is_thu", "is_fri", "is_sat"]:
        feats[c] = feats[c].fillna(0).astype(np.int8)

    feats["src_cart"] = (feats["src_code"] == 3).astype(np.int8)
    feats["src_repeat"] = (feats["src_code"] == 2).astype(np.int8)
    feats["src_recent"] = (feats["src_code"] == 1).astype(np.int8)
    feats["src_popular"] = (feats["src_code"] == 0).astype(np.int8)
    feats["src_priority"] = feats["src_code"].astype(np.float32)

    feats["v5_score"] = (
        feats["src_priority"]
        * feats["price_bonus"]
        * (1.0 + 0.5 * feats["item_purchase_rate"])
        + 0.05 * np.log1p(feats["item_view_pop"].astype(np.float32))
    ).astype(np.float32)

    # Cluster features
    if item_profile is not None and "cluster_id" in item_profile.columns:
        logging.info("[Feature] Adding cluster features...")
        cluster_map = item_profile.set_index("item_id")[["cluster_id", "avg_views_per_user", "view_to_purchase_rate", "fine_category"]]
        feats = feats.merge(
            cluster_map.reset_index()[["item_id", "cluster_id", "avg_views_per_user", "view_to_purchase_rate", "fine_category"]],
            on="item_id",
            how="left",
            suffixes=("", "_profile")
        )
        feats["item_cluster_id"] = feats["cluster_id"].fillna(-1).astype(np.int16)
        feats["item_avg_views_per_user"] = feats["avg_views_per_user"].fillna(1.0).astype(np.float32)
        feats["item_profile_conversion"] = feats["view_to_purchase_rate"].fillna(0.0).astype(np.float32)
        feats["item_fine_category"] = feats["fine_category"].fillna("other_c-1")

        n_missing_cluster = (feats["item_cluster_id"] == -1).sum()
        logging.info(f"  item_cluster_id: missing={n_missing_cluster} ({n_missing_cluster/len(feats)*100:.2f}%)")

        if user_view_map is not None:
            feats = _add_cluster_match_features(feats, item_profile, user_view_map)
        else:
            feats["cluster_match_score"] = 0.0
            feats["fine_category_match_score"] = 0.0
    else:
        feats["item_cluster_id"] = -1
        feats["item_avg_views_per_user"] = 1.0
        feats["item_profile_conversion"] = 0.0
        feats["cluster_match_score"] = 0.0
        feats["fine_category_match_score"] = 0.0

    X = feats[FEATURE_COLS]
    pairs = feats[["user_id", "item_id"]].copy()
    return X, pairs, groups


def _add_cluster_match_features(feats, item_profile, user_view_map):
    """Add cluster and fine_category match scores"""
    logging.info("  Computing cluster_match_score...")

    item_to_cluster = item_profile.set_index("item_id")["cluster_id"].to_dict()

    unique_users = feats["user_id"].unique()
    user_cluster_affinity_local = {}
    user_view_count = {}

    for u in unique_users:
        user_views = user_view_map.get(u, [])
        n_views = len(user_views)
        user_view_count[u] = n_views

        if n_views == 0:
            continue
        cluster_counts = defaultdict(int)
        for item in user_views:
            c = item_to_cluster.get(item, -1)
            if c != -1:
                cluster_counts[c] += 1
        if cluster_counts:
            total = sum(cluster_counts.values())
            user_cluster_affinity_local[u] = {c: cnt / total for c, cnt in cluster_counts.items()}

    logging.info(f"    Pre-computed affinity for {len(user_cluster_affinity_local):,} users")

    affinity_records = []
    for u, aff_dict in user_cluster_affinity_local.items():
        n_views = user_view_count.get(u, 0)
        view_penalty = min(n_views / MIN_VIEWS_FOR_FULL_SCORE, 1.0)

        for c, score in aff_dict.items():
            penalized_score = score * view_penalty
            affinity_records.append((u, c, penalized_score))

    if affinity_records:
        affinity_df = pd.DataFrame(affinity_records, columns=["user_id", "cluster_id", "affinity_score"])
        feats = feats.merge(
            affinity_df,
            left_on=["user_id", "item_cluster_id"],
            right_on=["user_id", "cluster_id"],
            how="left"
        )
        feats["cluster_match_score"] = feats["affinity_score"].fillna(0.0).astype(np.float32)
        feats = feats.drop(columns=["cluster_id", "affinity_score"], errors="ignore")
    else:
        feats["cluster_match_score"] = 0.0

    logging.info(f"  cluster_match_score: mean={feats['cluster_match_score'].mean():.4f}")

    # fine_category_match_score
    if "fine_category" in item_profile.columns:
        logging.info("  Computing fine_category_match_score...")

        item_to_fine_category = item_profile.set_index("item_id")["fine_category"].to_dict()

        user_fine_category_affinity_local = {}
        for u in unique_users:
            user_views = user_view_map.get(u, [])
            if len(user_views) == 0:
                continue
            fine_cat_counts = defaultdict(int)
            for item in user_views:
                fine_cat = item_to_fine_category.get(item, "other_c-1")
                if fine_cat and fine_cat != "other_c-1":
                    fine_cat_counts[fine_cat] += 1
            if fine_cat_counts:
                total = sum(fine_cat_counts.values())
                user_fine_category_affinity_local[u] = {cat: cnt / total for cat, cnt in fine_cat_counts.items()}

        fine_cat_affinity_records = []
        for u, aff_dict in user_fine_category_affinity_local.items():
            n_views = user_view_count.get(u, 0)
            view_penalty = min(n_views / MIN_VIEWS_FOR_FULL_SCORE, 1.0)

            for cat, score in aff_dict.items():
                penalized_score = score * view_penalty
                fine_cat_affinity_records.append((u, cat, penalized_score))

        if fine_cat_affinity_records:
            fine_cat_affinity_df = pd.DataFrame(fine_cat_affinity_records, columns=["user_id", "fine_category", "fine_cat_affinity_score"])
            feats = feats.merge(
                fine_cat_affinity_df,
                left_on=["user_id", "item_fine_category"],
                right_on=["user_id", "fine_category"],
                how="left"
            )
            feats["fine_category_match_score"] = feats["fine_cat_affinity_score"].fillna(0.0).astype(np.float32)
            feats = feats.drop(columns=["fine_category", "fine_cat_affinity_score", "item_fine_category"], errors="ignore")
        else:
            feats["fine_category_match_score"] = 0.0

        logging.info(f"  fine_category_match_score: mean={feats['fine_category_match_score'].mean():.4f}")
    else:
        feats["fine_category_match_score"] = 0.0

    return feats


def build_feature_df_fast(cand_dict, src_dict, ui_mi, item_mi, purchased_mi=None,
                          item_cluster_map=None, user_cluster_affinity=None,
                          item_fine_category_map=None, user_fine_category_affinity=None):
    """
    Fast feature generation for inference
    Uses pre-computed MultiIndex lookups for efficiency
    """
    users = list(cand_dict.keys())

    all_user_ids = []
    all_item_ids = []
    all_src = []
    groups = []

    for u in users:
        items = cand_dict[u]
        srcs = src_dict[u]
        groups.append(len(items))
        all_user_ids.extend([u] * len(items))
        all_item_ids.extend(items)
        all_src.extend(srcs)

    pairs = pd.DataFrame({"user_id": all_user_ids, "item_id": all_item_ids})
    src_code = np.array(all_src, dtype=np.int8)

    if len(pairs) == 0:
        X = pd.DataFrame()
        pairs["v5_score"] = []
        return X, pairs, groups

    mi = pd.MultiIndex.from_frame(pairs[["user_id", "item_id"]])
    ui_feat = ui_mi.reindex(mi)

    ui_defaults = {
        "ui_view_cnt": 0, "ui_view_cnt_40h": 0, "ui_cart_flag": 0,
        "ui_last_hours_ago": 9999.0, "ui_last_dow": -1, "ui_last_hour": -1,
        "is_peak": 0, "is_active": 0, "is_thu": 0, "is_fri": 0, "is_sat": 0,
    }
    for c, v in ui_defaults.items():
        if c in ui_feat.columns:
            ui_feat[c] = ui_feat[c].fillna(v)
        else:
            ui_feat[c] = v

    vc = ui_feat["ui_view_cnt"].to_numpy()
    repeat2 = (vc >= 2).astype(np.int8)
    repeat3 = (vc >= 3).astype(np.int8)
    repeat5 = (vc >= 5).astype(np.int8)

    item_feat = item_mi.reindex(pairs["item_id"].values)
    item_defaults = {
        "item_view_pop": 0, "item_purchase_pop": 0, "item_purchase_rate": 0.0,
        "item_price": -1.0, "price_bucket": -1, "price_bonus": 1.0,
    }
    for c, v in item_defaults.items():
        if c in item_feat.columns:
            item_feat[c] = item_feat[c].fillna(v)
        else:
            item_feat[c] = v

    src_cart = (src_code == 3).astype(np.int8)
    src_repeat = (src_code == 2).astype(np.int8)
    src_recent = (src_code == 1).astype(np.int8)
    src_popular = (src_code == 0).astype(np.int8)
    src_priority = src_code.astype(np.float32)

    v5_score = (
        src_priority
        * item_feat["price_bonus"].to_numpy(dtype=np.float32)
        * (1.0 + 0.5 * item_feat["item_purchase_rate"].to_numpy(dtype=np.float32))
        + 0.05 * np.log1p(item_feat["item_view_pop"].to_numpy(dtype=np.float32))
    ).astype(np.float32)

    item_ids = pairs["item_id"].values
    user_ids = pairs["user_id"].values

    if item_cluster_map is not None:
        item_cluster_ids = np.array([
            item_cluster_map.get(i, (-1, 1.0, 0.0))[0] for i in item_ids
        ], dtype=np.int16)
        item_avg_views = np.array([
            item_cluster_map.get(i, (-1, 1.0, 0.0))[1] for i in item_ids
        ], dtype=np.float32)
        item_profile_conv = np.array([
            item_cluster_map.get(i, (-1, 1.0, 0.0))[2] for i in item_ids
        ], dtype=np.float32)

        if user_cluster_affinity is not None:
            cluster_match = np.array([
                user_cluster_affinity.get(u, {}).get(c, 0.0)
                for u, c in zip(user_ids, item_cluster_ids)
            ], dtype=np.float32)
        else:
            cluster_match = np.zeros(len(pairs), dtype=np.float32)
    else:
        item_cluster_ids = np.full(len(pairs), -1, dtype=np.int16)
        item_avg_views = np.ones(len(pairs), dtype=np.float32)
        item_profile_conv = np.zeros(len(pairs), dtype=np.float32)
        cluster_match = np.zeros(len(pairs), dtype=np.float32)

    # fine_category features
    if item_fine_category_map is not None and user_fine_category_affinity is not None:
        item_fine_categories = [item_fine_category_map.get(i, "other_c-1") for i in item_ids]

        fine_category_match = np.array([
            user_fine_category_affinity.get(u, {}).get(fine_cat, 0.0)
            for u, fine_cat in zip(user_ids, item_fine_categories)
        ], dtype=np.float32)
    else:
        fine_category_match = np.zeros(len(pairs), dtype=np.float32)

    X = pd.DataFrame({
        "ui_view_cnt": ui_feat["ui_view_cnt"].to_numpy(dtype=np.int16),
        "ui_view_cnt_40h": ui_feat["ui_view_cnt_40h"].to_numpy(dtype=np.int16),
        "repeat2": repeat2, "repeat3": repeat3, "repeat5": repeat5,
        "ui_cart_flag": ui_feat["ui_cart_flag"].to_numpy(dtype=np.int8),
        "ui_last_hours_ago": ui_feat["ui_last_hours_ago"].to_numpy(dtype=np.float32),
        "ui_last_dow": ui_feat["ui_last_dow"].to_numpy(dtype=np.int8),
        "ui_last_hour": ui_feat["ui_last_hour"].to_numpy(dtype=np.int8),
        "is_peak": ui_feat["is_peak"].to_numpy(dtype=np.int8),
        "is_active": ui_feat["is_active"].to_numpy(dtype=np.int8),
        "is_thu": ui_feat["is_thu"].to_numpy(dtype=np.int8),
        "is_fri": ui_feat["is_fri"].to_numpy(dtype=np.int8),
        "is_sat": ui_feat["is_sat"].to_numpy(dtype=np.int8),
        "item_view_pop": item_feat["item_view_pop"].to_numpy(dtype=np.int32),
        "item_purchase_pop": item_feat["item_purchase_pop"].to_numpy(dtype=np.int32),
        "item_purchase_rate": item_feat["item_purchase_rate"].to_numpy(dtype=np.float32),
        "item_price": item_feat["item_price"].to_numpy(dtype=np.float32),
        "price_bucket": item_feat["price_bucket"].to_numpy(dtype=np.int16),
        "price_bonus": item_feat["price_bonus"].to_numpy(dtype=np.float32),
        "src_cart": src_cart,
        "src_repeat": src_repeat,
        "src_recent": src_recent,
        "src_popular": src_popular,
        "src_priority": src_priority,
        "v5_score": v5_score,
        "item_cluster_id": item_cluster_ids,
        "item_avg_views_per_user": item_avg_views,
        "item_profile_conversion": item_profile_conv,
        "cluster_match_score": cluster_match,
        "fine_category_match_score": fine_category_match,
    })[FEATURE_COLS]

    pairs["v5_score"] = v5_score

    return X, pairs, groups
