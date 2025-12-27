# inference.py
# Inference and recommendation for XGBoost GPU Learning to Rank v26

import logging
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict

from config import (
    BATCH_SIZE, RECENT_DAYS, BLEND_ALPHA, USE_RANK_BLEND,
    MIN_VIEWS_FOR_FULL_SCORE
)
from item_profile import get_user_cluster_affinity


def recommend_for_cold_user(user_id, popular_recent, item_stats, purchased_set, K=10):
    """
    Recommendation for cold users (no activity)

    Strategy:
    1. Recent trending items
    2. High conversion + low price items as fallback
    """
    out = []
    seen = set()
    bought = purchased_set.get(user_id, set())

    # Try recent trending items first
    for i in popular_recent:
        if i in bought or i in seen:
            continue
        out.append(i)
        seen.add(i)
        if len(out) >= K:
            return out

    # Fallback: high conversion + low price
    cand = (
        item_stats
        .sort_values(["item_purchase_rate", "item_price"], ascending=[False, True])
        ["item_id"]
        .tolist()
    )

    for i in cand:
        if i in bought or i in seen:
            continue
        out.append(i)
        if len(out) >= K:
            break

    return out


def get_recent_trending_items(df, recent_days=RECENT_DAYS, top_k=500):
    """Get recent trending items based on view counts"""
    T_recent = df["event_time"].max() - pd.Timedelta(days=recent_days)
    popular_recent = (
        df[df["event_time"] >= T_recent]
        .groupby("item_id")["user_id"]
        .count()
        .sort_values(ascending=False)
        .head(top_k)
        .index
        .tolist()
    )
    logging.info(f"[Infer] Recent trending items (last {recent_days} days): {len(popular_recent)}")
    return popular_recent


def compute_user_cluster_affinity(user_view_map, item_profile):
    """
    Pre-compute user cluster affinity with view penalty

    Low-view users get penalized scores to reduce overfitting
    """
    logging.info("[Infer] Pre-computing user cluster affinity (with view penalty)...")
    user_cluster_affinity = {}

    for user_id, view_history in user_view_map.items():
        affinity = get_user_cluster_affinity(user_id, view_history, item_profile)
        if affinity:
            n_views = len(view_history)
            view_penalty = min(n_views / MIN_VIEWS_FOR_FULL_SCORE, 1.0)
            if view_penalty < 1.0:
                affinity = {c: score * view_penalty for c, score in affinity.items()}
            user_cluster_affinity[user_id] = affinity

    logging.info(f"  User cluster affinity computed for {len(user_cluster_affinity):,} users")
    return user_cluster_affinity


def compute_user_fine_category_affinity(user_view_map, item_fine_category_map):
    """
    Pre-compute user fine_category affinity with view penalty
    """
    logging.info("[Infer] Pre-computing user fine_category affinity...")
    user_fine_category_affinity = {}

    for user_id, view_history in user_view_map.items():
        n_views = len(view_history)
        if n_views == 0:
            continue

        fine_cat_counts = defaultdict(int)
        for item in view_history:
            fine_cat = item_fine_category_map.get(item, "other_c-1")
            if fine_cat and fine_cat != "other_c-1":
                fine_cat_counts[fine_cat] += 1

        if fine_cat_counts:
            total = sum(fine_cat_counts.values())
            view_penalty = min(n_views / MIN_VIEWS_FOR_FULL_SCORE, 1.0)
            user_fine_category_affinity[user_id] = {
                cat: (cnt / total) * view_penalty for cat, cnt in fine_cat_counts.items()
            }

    logging.info(f"  User fine_category affinity computed for {len(user_fine_category_affinity):,} users")
    return user_fine_category_affinity


def run_batch_inference(
    target_users,
    cand_all,
    src_all,
    ranker,
    best_iter,
    ui_indexed,
    item_stats_indexed,
    purchased_set,
    purchased_map,
    item_stats,
    item_cluster_map,
    user_cluster_affinity,
    item_fine_category_map,
    user_fine_category_affinity,
    user_activity,
    popular_recent,
    popular_items,
    build_feature_df_fast,
):
    """
    Run batch inference for all target users

    Args:
        target_users: List of target user IDs
        cand_all: Dict of candidate items per user
        src_all: Dict of source codes per user
        ranker: Trained XGBoost model
        best_iter: Best iteration for prediction
        ... (other lookup dictionaries)
        build_feature_df_fast: Feature building function

    Returns:
        all_recs: List of recommendation dicts
        cold_user_count: Number of cold users
        fallback_count: Number of users needing fallback
    """
    all_recs = []
    n_batches = (len(target_users) + BATCH_SIZE - 1) // BATCH_SIZE
    logging.info(f"[Infer] processing {len(target_users):,} users in {n_batches} batches.")

    popular_items_list = list(popular_items)
    logging.info(f"[Infer] Blend settings: ALPHA={BLEND_ALPHA}, RANK_BLEND={USE_RANK_BLEND}")

    fallback_count = 0
    cold_user_count = 0

    for batch_start in range(0, len(target_users), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(target_users))
        batch_users = target_users[batch_start:batch_end]
        batch_num = batch_start // BATCH_SIZE + 1

        logging.info(f"  Batch {batch_num}/{n_batches}: users {batch_start:,}-{batch_end:,}")

        cand_batch = {u: cand_all.get(u, []) for u in batch_users}
        src_batch = {u: src_all.get(u, [0] * len(cand_batch[u])) for u in batch_users}

        # Build features
        Xte, pte, gte = build_feature_df_fast(
            cand_batch, src_batch, ui_indexed, item_stats_indexed, purchased_mi=None,
            item_cluster_map=item_cluster_map, user_cluster_affinity=user_cluster_affinity,
            item_fine_category_map=item_fine_category_map, user_fine_category_affinity=user_fine_category_affinity
        )

        # XGBoost prediction
        dtest = xgb.DMatrix(Xte)
        yscore = ranker.predict(dtest, iteration_range=(0, best_iter))

        # Blend XGBoost score + v5_score
        EPS = 1e-9
        pte = pte.copy()
        pte["xgb_score"] = yscore.astype(np.float32)

        if "v5_score" not in pte.columns:
            raise RuntimeError("pte has no 'v5_score'. Check build_feature_df_fast().")

        if USE_RANK_BLEND:
            pte["xgb_rank"] = pte.groupby("user_id")["xgb_score"].rank(method="average", pct=True)
            pte["v5_rank"] = pte.groupby("user_id")["v5_score"].rank(method="average", pct=True)
            pte["score"] = (BLEND_ALPHA * pte["xgb_rank"] + (1.0 - BLEND_ALPHA) * pte["v5_rank"]).astype(np.float32)
            pte.drop(columns=["xgb_rank", "v5_rank"], inplace=True)
        else:
            l = pte["xgb_score"].to_numpy(dtype=np.float32)
            v = pte["v5_score"].to_numpy(dtype=np.float32)
            l = (l - l.mean()) / (l.std() + EPS)
            v = (v - v.mean()) / (v.std() + EPS)
            pte["score"] = (BLEND_ALPHA * l + (1.0 - BLEND_ALPHA) * v).astype(np.float32)

        # Exclude purchased items
        mi_pte = pd.MultiIndex.from_frame(pte[["user_id", "item_id"]])
        mask = ~mi_pte.isin(purchased_set)
        pte = pte.loc[mask].copy()

        pte = pte.drop_duplicates(["user_id", "item_id"], keep="first")

        pte_sorted = pte.sort_values(["user_id", "score"], ascending=[True, False])
        topk_df = pte_sorted.groupby("user_id", sort=False).head(10)
        topk_map = topk_df.groupby("user_id")["item_id"].apply(list).to_dict()

        # Slot-based final assembly
        for u in batch_users:
            recs = topk_map.get(u, [])
            activity = user_activity.get(u, 0)

            if activity <= 0:
                cold_user_count += 1
                cold_recs = recommend_for_cold_user(
                    u,
                    popular_recent,
                    item_stats,
                    purchased_map,
                    K=10
                )
                for i in cold_recs:
                    all_recs.append({"user_id": u, "item_id": i})
                continue

            seen = set()
            for i in recs:
                if i not in seen:
                    all_recs.append({"user_id": u, "item_id": i})
                    seen.add(i)

            if len(seen) < 10:
                fallback_count += 1
                for pi in popular_items_list:
                    if pi in purchased_map.get(u, set()) or pi in seen:
                        continue
                    all_recs.append({"user_id": u, "item_id": pi})
                    seen.add(pi)
                    if len(seen) >= 10:
                        break

        del Xte, pte, gte, pte_sorted, topk_df, cand_batch
        gc.collect()

    return all_recs, cold_user_count, fallback_count


def log_output_stats(sub, cold_user_count, fallback_count):
    """Log output statistics"""
    n_users_in_sub = sub["user_id"].nunique()
    n_items_in_sub = sub["item_id"].nunique()
    recs_per_user = sub.groupby("user_id").size()

    logging.info(f"[Output] Total recs: {len(sub):,}")
    logging.info(f"[Output] Users: {n_users_in_sub:,}")
    logging.info(f"[Output] Unique items: {n_items_in_sub:,}")
    logging.info(f"[Output] Cold users (activity<=0): {cold_user_count:,} ({cold_user_count/n_users_in_sub*100:.1f}%)")
    logging.info(f"[Output] Fallback users: {fallback_count:,}")
    logging.info(f"[Output] Recs per user: min={recs_per_user.min()}, max={recs_per_user.max()}, avg={recs_per_user.mean():.2f}")

    # Top recommended items distribution
    item_counts = sub["item_id"].value_counts()
    logging.info("[Output] Top 10 recommended items:")
    for i, (item, cnt) in enumerate(item_counts.head(10).items()):
        pct = cnt / n_users_in_sub * 100
        logging.info(f"  {i+1}. {str(item)[:40]}: {cnt:,} ({pct:.2f}%)")

    return {
        "total_recs": len(sub),
        "n_users": n_users_in_sub,
        "n_unique_items": n_items_in_sub,
        "cold_users": cold_user_count,
        "fallback_users": fallback_count,
        "avg_recs_per_user": float(recs_per_user.mean()),
    }
