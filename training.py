# training.py
# XGBoost GPU Learning to Rank training for v26

import logging
import xgboost as xgb

from config import XGB_GPU_PARAMS, EARLY_STOPPING_ROUNDS


def train_ranker_xgb(X_train, y_train, group_train, X_val, y_val, group_val):
    """
    Train XGBoost GPU Learning to Rank model

    Uses NDCG@10 as objective and evaluation metric.
    Optimized for RTX 3090 24GB VRAM.

    Args:
        X_train: Training features DataFrame
        y_train: Training labels (implicit weighted: purchase=5, cart=3, view=1)
        group_train: Group sizes for training (users)
        X_val: Validation features DataFrame
        y_val: Validation labels
        group_val: Group sizes for validation

    Returns:
        bst: Trained XGBoost Booster
        best_iter: Best iteration number
    """
    logging.info("[Train] Starting XGBoost GPU training...")
    logging.info(f"  GPU parameters: device={XGB_GPU_PARAMS['device']}, tree_method={XGB_GPU_PARAMS['tree_method']}")
    logging.info(f"  Tree parameters: max_depth={XGB_GPU_PARAMS['max_depth']}, max_leaves={XGB_GPU_PARAMS['max_leaves']}")
    logging.info(f"  Learning parameters: lr={XGB_GPU_PARAMS['learning_rate']}, n_estimators={XGB_GPU_PARAMS['n_estimators']}")

    # Create DMatrix for Learning to Rank
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(group_train)

    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group(group_val)

    # XGBoost parameters
    params = {
        "device": XGB_GPU_PARAMS["device"],
        "tree_method": XGB_GPU_PARAMS["tree_method"],
        "objective": XGB_GPU_PARAMS["objective"],
        "eval_metric": XGB_GPU_PARAMS["eval_metric"],
        "eta": XGB_GPU_PARAMS["learning_rate"],
        "max_depth": XGB_GPU_PARAMS["max_depth"],
        "max_leaves": XGB_GPU_PARAMS["max_leaves"],
        "min_child_weight": XGB_GPU_PARAMS["min_child_weight"],
        "reg_alpha": XGB_GPU_PARAMS["reg_alpha"],
        "reg_lambda": XGB_GPU_PARAMS["reg_lambda"],
        "gamma": XGB_GPU_PARAMS["gamma"],
        "subsample": XGB_GPU_PARAMS["subsample"],
        "colsample_bytree": XGB_GPU_PARAMS["colsample_bytree"],
        "colsample_bylevel": XGB_GPU_PARAMS["colsample_bylevel"],
        "max_bin": XGB_GPU_PARAMS["max_bin"],
        "seed": XGB_GPU_PARAMS["seed"],
        "verbosity": XGB_GPU_PARAMS["verbosity"],
    }

    evals = [(dtrain, "train"), (dval, "eval")]

    # Train
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=XGB_GPU_PARAMS["n_estimators"],
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=100
    )

    best_iter = bst.best_iteration
    logging.info(f"[Train] best_iteration={best_iter}")

    return bst, best_iter


def log_feature_importance(ranker, top_k=10):
    """Log top feature importances"""
    importance = ranker.get_score(importance_type='gain')
    logging.info(f"[Train] Feature Importance (Top {top_k}):")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for fname, fval in sorted_imp[:top_k]:
        logging.info(f"  {fname}: {fval:.2f}")
    return importance
