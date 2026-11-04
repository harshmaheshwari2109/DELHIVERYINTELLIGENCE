import os
import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 1. Load data
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Record metrics before cleaning for checks
raw_rows = len(raw)
dup_before = raw.duplicated().sum()
missing_dist_before = raw["distance_km"].isnull().sum()
missing_weight_before = raw["parcel_weight_kg"].isnull().sum()
sentinel_weight = (raw["parcel_weight_kg"] > 40).sum()

# Clean raw data
cleaned_df, medians = clean_shipments(raw)
rows_after_dedup = len(cleaned_df)

# Build features and modeling table
mod_table = build_features(cleaned_df, hubs)
mod_table["is_delayed"] = (
    mod_table["delivery_hours"] > mod_table["promised_hours"]
).astype(int)

mod_table.to_csv("data/modelling_table.csv", index=False)

# Save Part A Checks
checks_data = [
    {"metric": "raw_rows", "value": raw_rows},
    {"metric": "duplicate_rows_removed", "value": dup_before},
    {"metric": "rows_after_dedup", "value": rows_after_dedup},
    {"metric": "missing_distance_km_before", "value": missing_dist_before},
    {
        "metric": "missing_parcel_weight_before",
        "value": missing_weight_before,
    },
    {"metric": "sentinel_weight_rows", "value": sentinel_weight},
    {"metric": "delayed_count", "value": mod_table["is_delayed"].sum()},
    {
        "metric": "delay_rate",
        "value": round(mod_table["is_delayed"].mean(), 4),
    },
]
pd.DataFrame(checks_data).to_csv("outputs/part_a_checks.csv", index=False)

# Delay Evidence Table by service_type, hub_tier, distance_bins
mod_table["distance_band"] = pd.cut(
    mod_table["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
)

evidence_list = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    grouped = (
        mod_table.groupby(dim, observed=False)
        .agg(
            shipments=("delivery_hours", "count"),
            delayed=("is_delayed", "sum"),
            mean_hours=("delivery_hours", "mean"),
        )
        .reset_index()
    )
    grouped.rename(columns={dim: "group"}, inplace=True)
    grouped["dimension"] = dim
    grouped["delay_rate"] = grouped["delayed"] / grouped["shipments"]
    evidence_list.append(
        grouped[
            [
                "dimension",
                "group",
                "shipments",
                "delayed",
                "delay_rate",
                "mean_hours",
            ]
        ]
    )

pd.concat(evidence_list, ignore_index=True).to_csv(
    "outputs/part_a_delay_by_group.csv", index=False
)

# Correlation matrix
corr_cols = FEATURES + ["delivery_hours", "is_delayed"]
corr_matrix = mod_table[corr_cols].corr()
corr_matrix.to_csv("outputs/part_a_correlations.csv")
print("Part A completed successfully!")