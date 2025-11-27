import os
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def run_shift_planning(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 1200,
    max_depth: int = 25,
    n_jobs: int = -1,
    save_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:

    df = df.copy()

    # ===== Validate required cols =====
    required_cols = {"INSPECTOR ID", "ZONES", "Inspection Type", "AREA", "Rush Score per 100"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    fine_cols = [col for col in df.columns if "total fines" in col.lower()]
    if not fine_cols:
        raise ValueError("No 'total fines' columns found.")

    # ===== Clean fines =====
    df[fine_cols] = df[fine_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["total_fines_6m"] = df[fine_cols].sum(axis=1)
    df["total_fines_log"] = np.log1p(df["total_fines_6m"])

    # ===== Features =====
    drop_cols = ["INSPECTOR ID", "total_fines_6m", "total_fines_log"]
    X = df.drop(columns=drop_cols)
    y = df["total_fines_log"]
    X = pd.get_dummies(X)

    # ===== Split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # ===== Train Model =====
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    mae = mean_absolute_error(y_test_real, y_pred)
    accuracy = (1 - mae / np.mean(y_test_real)) * 100

    metrics = {"mae": float(mae), "accuracy_pct": float(accuracy)}

    # ===== Ranking =====
    df["Rank (Zone & Type)"] = (
        df.groupby(["ZONES", "Inspection Type"])["total_fines_6m"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    # ===== Area Ranking =====
    areas_rank = df[["ZONES", "Inspection Type", "AREA", "Rush Score per 100"]].drop_duplicates()
    areas_rank["Rush Score per 100"] = pd.to_numeric(
        areas_rank["Rush Score per 100"], errors="coerce").fillna(0)
    areas_rank = areas_rank.sort_values(by="Rush Score per 100", ascending=False)

    # ===== Assign Areas =====
    assignments = []
    for (zone, type_group), group in df.groupby(["ZONES", "Inspection Type"]):
        sorted_insp = group.sort_values("Rank (Zone & Type)")

        area_list = areas_rank[
            (areas_rank["ZONES"] == zone)
            & (areas_rank["Inspection Type"] == type_group)
        ]["AREA"].tolist()

        if not area_list:
            assigned = ["No Area"] * len(sorted_insp)
        else:
            repeat = (len(sorted_insp) // len(area_list)) + 1
            assigned = (area_list * repeat)[: len(sorted_insp)]

        temp = pd.DataFrame({
            "INSPECTOR ID": sorted_insp["INSPECTOR ID"],
            "ZONES": zone,
            "Inspection Type": type_group,
            "Assigned Area (High Rush)": assigned,
        })
        assignments.append(temp)

    assignments_df = pd.concat(assignments, ignore_index=True)

    # ===== Final Output =====
    final_df = df.merge(assignments_df, on=["INSPECTOR ID", "ZONES", "Inspection Type"])
    final_df = final_df.drop(columns=fine_cols + ["total_fines_log"])

    if save_path:
        final_df.to_excel(save_path, index=False)

    return final_df, metrics
