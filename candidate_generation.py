# candidate_generation.py
# Candidate generation for XGBoost GPU Learning to Rank v26

import logging
import numpy as np
import pandas as pd

from config import CAND_CART, CAND_REPEAT, CAND_RECENT, CAND_MAX


def build_candidates(df, target_users, ui, popular_items):
    """
    Generate candidate items for target users

    Candidate sources (priority order):
    1. Cart items (src_code=3)
    2. Repeat viewed items (src_code=2)
    3. Recent viewed items (src_code=1)
    4. Popular items (src_code=0)
    """
    logging.info("[Cand] generating candidates for target users...")
    target_set = set(target_users)

    # Recent unique views
    view = df[df["event_type"] == "view"][["user_id", "item_id", "event_time"]].copy()
    view = view[view["user_id"].isin(target_set)]
    view = view.sort_values("event_time", ascending=False)
    view = view.drop_duplicates(["user_id", "item_id"], keep="first")
    view["rank"] = view.groupby("user_id").cumcount()
    recent_unique = view[view["rank"] < (CAND_REPEAT + CAND_RECENT + 50)][["user_id", "item_id"]]

    # Cart items
    cart = df[df["event_type"] == "cart"][["user_id", "item_id", "event_time"]].copy()
    cart = cart[cart["user_id"].isin(target_set)]
    cart = cart.sort_values("event_time", ascending=False)
    cart = cart.drop_duplicates(["user_id", "item_id"], keep="first")
    cart["rank"] = cart.groupby("user_id").cumcount()
    cart_top = cart[cart["rank"] < CAND_CART][["user_id", "item_id"]]

    # Repeat viewed items (viewed >= 2 times)
    ui_small = ui[ui["user_id"].isin(target_set)][["user_id", "item_id", "ui_view_cnt"]]
    repeat = recent_unique.merge(ui_small, on=["user_id", "item_id"], how="left")
    repeat = repeat[repeat["ui_view_cnt"].fillna(0) >= 2][["user_id", "item_id"]]
    repeat["rank"] = repeat.groupby("user_id").cumcount()
    repeat_top = repeat[repeat["rank"] < CAND_REPEAT][["user_id", "item_id"]]

    # Recent viewed items
    recent_unique["rank"] = recent_unique.groupby("user_id").cumcount()
    recent_top = recent_unique[recent_unique["rank"] < CAND_RECENT][["user_id", "item_id"]]

    # Build mappings
    cart_map = cart_top.groupby("user_id")["item_id"].apply(list).to_dict()
    repeat_map = repeat_top.groupby("user_id")["item_id"].apply(list).to_dict()
    recent_map = recent_top.groupby("user_id")["item_id"].apply(list).to_dict()

    cand_dict = {}
    src_dict = {}

    for u in target_users:
        seen = set()
        cand = []
        srcs = []

        # 1. Cart items (priority 3)
        for i in cart_map.get(u, []):
            if i in seen:
                continue
            cand.append(i)
            srcs.append(3)
            seen.add(i)
            if len(cand) >= CAND_MAX:
                break

        # 2. Repeat viewed items (priority 2)
        if len(cand) < CAND_MAX:
            for i in repeat_map.get(u, []):
                if i in seen:
                    continue
                cand.append(i)
                srcs.append(2)
                seen.add(i)
                if len(cand) >= CAND_MAX:
                    break

        # 3. Recent viewed items (priority 1)
        if len(cand) < CAND_MAX:
            for i in recent_map.get(u, []):
                if i in seen:
                    continue
                cand.append(i)
                srcs.append(1)
                seen.add(i)
                if len(cand) >= CAND_MAX:
                    break

        # 4. Popular items (priority 0)
        if len(cand) < CAND_MAX:
            for i in popular_items:
                if i in seen:
                    continue
                cand.append(i)
                srcs.append(0)
                seen.add(i)
                if len(cand) >= CAND_MAX:
                    break

        cand_dict[u] = cand
        src_dict[u] = srcs

    logging.info("  done.")
    return cand_dict, src_dict
