# main.py
# XGBoost GPU Learning to Rank v26 - fine_category (real_category + cluster)
# Main entry point

import gc
import json
import logging
import numpy as np
import pandas as pd

from config import (
    OUT_DIR, OUT_PATH, LOG_PATH, ANALYSIS_PATH, TIMESTAMP,
    VAL_FUTURE_DAYS, MAX_TRAIN_USERS, N_CLUSTERS,
    XGB_GPU_PARAMS, EVENT_WEIGHTS
)
from utils import (
    setup_logging, safe_mkdir, split_users_hash,
    filter_empty_groups, make_weighted_label
)
from data_loader import (
    load_data, build_global_stats, build_user_item_agg,
    prepare_indexed_lookups
)
from item_profile import (
    build_item_behavior_profile, cluster_items_by_behavior,
    build_cluster_top_items, build_brand_cluster_map,
    build_item_cluster_map, build_item_fine_category_map
)
from candidate_generation import build_candidates
from feature_engineering import build_feature_df, build_feature_df_fast
from training import train_ranker_xgb, log_feature_importance
from inference import (
    get_recent_trending_items,
    compute_user_cluster_affinity,
    compute_user_fine_category_affinity,
    run_batch_inference,
    log_output_stats
)


def main():
    logger = setup_logging()

    # Analysis dictionary for logging
    analysis = {
        "timestamp": TIMESTAMP,
        "version": "v26_fine_category_xgboost_gpu",
        "model": "XGBoost GPU",
        "gpu_config": XGB_GPU_PARAMS,
        "config": {
            "VAL_FUTURE_DAYS": VAL_FUTURE_DAYS,
            "MAX_TRAIN_USERS": MAX_TRAIN_USERS,
        },
        "data_stats": {},
        "training_stats": {},
        "inference_stats": {},
        "output_stats": {},
    }

    logging.info("=" * 60)
    logging.info("XGBoost GPU Rerank v26 - fine_category (real_category + cluster)")
    logging.info("=" * 60)
    logging.info("✅ Key features:")
    logging.info("  1. XGBoost GPU acceleration (optimized for RTX 3090 24GB)")
    logging.info("  2. fine_category = real_category + cluster_id")
    logging.info("  3. Learning to Rank with NDCG@10 objective")
    logging.info("  4. All users go through XGBoost ranking (no special handling for low-view users)")
    logging.info("=" * 60)
    logging.info(f"Output: {OUT_PATH}")
    logging.info(f"Log: {LOG_PATH}")
    logging.info(f"Analysis: {ANALYSIS_PATH}")

    safe_mkdir(OUT_DIR)

    # =========================
    # 1. Load Data
    # =========================
    df, sample, target_users = load_data()

    # Time split
    T_end = df["event_time"].max()
    T_cut = T_end - pd.Timedelta(days=VAL_FUTURE_DAYS)
    past_logs = df[df["event_time"] < T_cut].copy()
    future_logs = df[df["event_time"] >= T_cut].copy()
    logging.info(f"[Split] past_rows={len(past_logs):,}, future_rows={len(future_logs):,}, T_cut={T_cut}")

    analysis["data_stats"]["total_rows"] = len(df)
    analysis["data_stats"]["past_rows"] = len(past_logs)
    analysis["data_stats"]["future_rows"] = len(future_logs)
    analysis["data_stats"]["target_users"] = len(target_users)

    # =========================
    # 2. Global Statistics
    # =========================
    item_stats, popular_items = build_global_stats(df)
    analysis["data_stats"]["n_items_with_views"] = len(item_stats)

    # =========================
    # 3. Item Behavior Profile & Clustering
    # =========================
    logging.info("=" * 60)
    logging.info("Building Item Behavior Profiles & Clustering (with fine_category)")
    logging.info("=" * 60)

    item_profile = build_item_behavior_profile(df)
    item_profile, cluster_stats, kmeans_model, scaler = cluster_items_by_behavior(
        item_profile, n_clusters=N_CLUSTERS
    )

    cluster_top_items = build_cluster_top_items(item_profile, item_stats, top_k=20)
    brand_cluster_map = build_brand_cluster_map(item_profile)

    # Merge cluster info to item_stats
    item_stats = item_stats.merge(
        item_profile[["item_id", "cluster_id", "avg_views_per_user", "view_to_purchase_rate"]],
        on="item_id",
        how="left"
    )
    item_stats["cluster_id"] = item_stats["cluster_id"].fillna(-1).astype(np.int16)

    # Build lookup maps
    item_cluster_map = build_item_cluster_map(item_profile)
    item_fine_category_map = build_item_fine_category_map(item_profile)
    n_fine_categories = len(set(item_fine_category_map.values()))
    logging.info(f"  item_fine_category_map built: {len(item_fine_category_map):,} items, {n_fine_categories} unique fine categories")

    analysis["data_stats"]["n_clusters"] = N_CLUSTERS
    analysis["data_stats"]["n_fine_categories"] = n_fine_categories

    # =========================
    # 4. User-Item Aggregates
    # =========================
    ui_past, max_time_past = build_user_item_agg(past_logs)

    # =========================
    # 5. Training Users Selection
    # =========================
    future_purchases = future_logs[future_logs["event_type"] == "purchase"][["user_id", "item_id"]].drop_duplicates()
    pos_users = future_purchases["user_id"].unique().tolist()

    if len(pos_users) > MAX_TRAIN_USERS:
        pos_users = pos_users[:MAX_TRAIN_USERS]

    active_users = past_logs[past_logs["event_type"] == "view"]["user_id"].dropna().unique().tolist()
    active_users = [u for u in active_users if u not in set(pos_users)]
    neg_add = max(0, min(len(active_users), MAX_TRAIN_USERS - len(pos_users)))
    if neg_add > 0:
        pos_users += active_users[:neg_add]

    train_users, val_users = split_users_hash(pos_users, valid_ratio=0.2, seed=42)
    logging.info(f"[Users] train_users={len(train_users):,}, val_users={len(val_users):,}")
    analysis["training_stats"]["train_users"] = len(train_users)
    analysis["training_stats"]["val_users"] = len(val_users)

    # =========================
    # 6. Candidate Generation
    # =========================
    cand_train, src_train = build_candidates(past_logs, train_users, ui_past, popular_items)
    cand_val, src_val = build_candidates(past_logs, val_users, ui_past, popular_items)

    past_purchases = past_logs[past_logs["event_type"] == "purchase"][["user_id", "item_id"]].drop_duplicates()

    # User view history for cluster/fine_category matching
    logging.info("[Feature] Building user view history for cluster/fine_category matching...")
    user_view_map = past_logs[past_logs["event_type"] == "view"].groupby("user_id")["item_id"].apply(list).to_dict()
    logging.info(f"  User view map built for {len(user_view_map):,} users")

    # =========================
    # 7. Feature Generation
    # =========================
    Xtr, ptr, gtr = build_feature_df(
        cand_train, src_train, ui_past, item_stats, max_time_past,
        exclude_purchased_pairs=past_purchases,
        item_profile=item_profile, user_view_map=user_view_map
    )
    Xva, pva, gva = build_feature_df(
        cand_val, src_val, ui_past, item_stats, max_time_past,
        exclude_purchased_pairs=past_purchases,
        item_profile=item_profile, user_view_map=user_view_map
    )

    # =========================
    # 8. Label Generation
    # =========================
    future_ev = future_logs[["user_id", "item_id", "event_type"]].drop_duplicates()
    future_label_map = {}
    for u, i, e in future_ev.itertuples(index=False):
        w = EVENT_WEIGHTS.get(e, 0)
        if w == 0:
            continue
        future_label_map[(u, i)] = max(future_label_map.get((u, i), 0), w)

    ytr = make_weighted_label(ptr, future_label_map)
    yva = make_weighted_label(pva, future_label_map)

    logging.info(f"[Label] train_nonzero={int((ytr>0).sum()):,}/{len(ytr):,}, val_nonzero={int((yva>0).sum()):,}/{len(yva):,}")
    analysis["training_stats"]["train_samples"] = len(ytr)
    analysis["training_stats"]["train_positives"] = int((ytr > 0).sum())
    analysis["training_stats"]["val_samples"] = len(yva)
    analysis["training_stats"]["val_positives"] = int((yva > 0).sum())

    # Filter empty groups
    Xtr, ptr, ytr, gtr, train_users_kept = filter_empty_groups(Xtr, ptr, ytr, gtr, list(cand_train.keys()))
    Xva, pva, yva, gva, val_users_kept = filter_empty_groups(Xva, pva, yva, gva, list(cand_val.keys()))

    logging.info(f"[Group] train_groups={len(gtr):,}, val_groups={len(gva):,}")
    analysis["training_stats"]["train_groups"] = len(gtr)
    analysis["training_stats"]["val_groups"] = len(gva)
    analysis["training_stats"]["feature_cols"] = list(Xtr.columns)

    # =========================
    # 9. XGBoost Training
    # =========================
    ranker, best_iter = train_ranker_xgb(Xtr, ytr, gtr, Xva, yva, gva)
    analysis["training_stats"]["best_iteration"] = best_iter

    # Feature importance
    importance = log_feature_importance(ranker)
    analysis["training_stats"]["feature_importance"] = importance

    # Free training memory
    del Xtr, Xva, ytr, yva, ptr, pva
    gc.collect()

    # =========================
    # 10. Inference Preparation
    # =========================
    logging.info("[Infer] build agg on full logs for inference.")
    ui_full, max_time_full = build_user_item_agg(df)

    popular_recent = get_recent_trending_items(df)

    purchased_pairs_full = df[df["event_type"] == "purchase"][["user_id", "item_id"]].drop_duplicates()
    purchased_map = purchased_pairs_full.groupby("user_id")["item_id"].apply(set).to_dict()

    # Full data user view history
    logging.info("[Infer] Building user view history for inference...")
    user_view_map_full = df[df["event_type"] == "view"].groupby("user_id")["item_id"].apply(list).to_dict()
    logging.info(f"  User view map (full) built for {len(user_view_map_full):,} users")

    # Pre-compute affinities
    user_cluster_affinity = compute_user_cluster_affinity(user_view_map_full, item_profile)
    user_fine_category_affinity = compute_user_fine_category_affinity(user_view_map_full, item_fine_category_map)

    # Indexed lookups
    logging.info("[Infer] preparing indexed lookups.")
    ui_indexed, item_stats_indexed, purchased_set = prepare_indexed_lookups(
        ui_full, item_stats, purchased_pairs_full
    )

    # Pre-compute candidates
    logging.info("[Infer] precomputing candidates ONCE for all target users...")
    df_min = df[df["event_type"].isin(["view", "cart"])][["user_id", "item_id", "event_time", "event_type"]].copy()
    cand_all, src_all = build_candidates(df_min, target_users, ui_full, popular_items)
    del df_min
    gc.collect()
    logging.info("[Infer] candidate precompute done.")

    total_cands = sum(len(v) for v in cand_all.values())
    avg_cands = total_cands / len(cand_all) if cand_all else 0
    logging.info(f"[Infer] Total candidates: {total_cands:,}, Avg per user: {avg_cands:.1f}")

    user_activity = ui_full.groupby("user_id")["ui_view_cnt"].sum()

    # =========================
    # 11. Run Inference
    # =========================
    all_recs, cold_user_count, fallback_count = run_batch_inference(
        target_users=target_users,
        cand_all=cand_all,
        src_all=src_all,
        ranker=ranker,
        best_iter=best_iter,
        ui_indexed=ui_indexed,
        item_stats_indexed=item_stats_indexed,
        purchased_set=purchased_set,
        purchased_map=purchased_map,
        item_stats=item_stats,
        item_cluster_map=item_cluster_map,
        user_cluster_affinity=user_cluster_affinity,
        item_fine_category_map=item_fine_category_map,
        user_fine_category_affinity=user_fine_category_affinity,
        user_activity=user_activity,
        popular_recent=popular_recent,
        popular_items=popular_items,
        build_feature_df_fast=build_feature_df_fast,
    )

    # =========================
    # 12. Build Submission
    # =========================
    logging.info("[Submit] build submission df.")
    sub = pd.DataFrame(all_recs)

    output_stats = log_output_stats(sub, cold_user_count, fallback_count)
    analysis["output_stats"] = output_stats

    # Save submission
    safe_mkdir(OUT_DIR)
    sub.to_csv(OUT_PATH, index=False)
    logging.info(f"[Done] Saved submission: {OUT_PATH}")

    # Save analysis JSON
    with open(ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    logging.info(f"[Done] Saved analysis: {ANALYSIS_PATH}")

    logging.info("=" * 60)
    logging.info("XGBoost GPU Rerank v26 Complete! (with fine_category)")
    logging.info("=" * 60)

    # Cleanup
    del df, past_logs, future_logs, ui_past, ui_full
    gc.collect()


if __name__ == "__main__":
    main()
